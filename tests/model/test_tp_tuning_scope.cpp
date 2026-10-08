// tests/model/test_tp_tuning_scope.cpp -- CPU-only checks of the GEMM tuning scope (docs/pp-tp2-hybrid.md 7, "Tuning scope"): the
// per-thread TP2 tuning flag (linear.h SetTp2TuningForThisThread) is thread state, but the right value is a property of the
// MODEL a call runs (a TP rank: true; a pipeline stage / TP=1 Model: false). The hybrid mode's rank worker owns a rank Model
// AND a stage Model, so every Model call runs inside a Tp2TuningScope with that Model's own value, and Model::CheckTuningScope
// (CheckTp2TuningScope) is the guard that fails loudly when the thread disagrees with the Model. This test covers the RAII
// semantics (set, restore, nesting, exceptions, thread isolation), the guard, and the negative controls: with Model::Load's
// set-on-every-load behaviour alone the flag follows the LAST load, so the first Model's calls silently pick the other
// layout's tunings -- the guard must fire for exactly that, and PickTuning must really return different rows for the two
// flag values on a rank-equal shape (gdn.in_proj_z's 6144 x 5120, which a TP rank's attn.qg also has). No HIP call: the
// tuning tables are host data; HIP_VISIBLE_DEVICES=-1 hides the devices from the runtime that linking r4dx_model_linear loads.
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

#include "linear.h"

using namespace r4dx::model;

namespace {

int g_fails = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fails;
  }
}

// What Model is for the guard: the value the flag must have for its calls, fixed at "load", which (as Model::Load does) also
// sets the thread's flag.
struct FakeModel {
  bool tp2;
  explicit FakeModel(bool is_tp_rank) : tp2(is_tp_rank) { SetTp2TuningForThisThread(is_tp_rank); }
  void Call() const { CheckTp2TuningScope(tp2, "FakeModel::Call"); }
  void ScopedCall() const {
    Tp2TuningScope s(tp2);
    Call();
  }
};

bool Throws(const FakeModel& m) {
  try {
    m.Call();
  } catch (const std::logic_error&) {
    return true;
  }
  return false;
}

bool SameTuning(const LinearTuning& a, const LinearTuning& b) {
  return a.WV == b.WV && a.SK == b.SK && a.MB == b.MB && a.NPW == b.NPW && a.NT == b.NT && a.SKG == b.SKG && a.U == b.U;
}

void Raii() {
  SetTp2TuningForThisThread(false);
  Check(!Tp2TuningForThisThread(), "the setter and the getter agree (false)");
  SetTp2TuningForThisThread(true);
  Check(Tp2TuningForThisThread(), "the setter and the getter agree (true)");
  SetTp2TuningForThisThread(false);

  {
    Tp2TuningScope s(true);
    Check(Tp2TuningForThisThread(), "a scope sets the flag");
  }
  Check(!Tp2TuningForThisThread(), "... and restores the previous value (false) on exit");
  SetTp2TuningForThisThread(true);
  {
    Tp2TuningScope s(false);
    Check(!Tp2TuningForThisThread(), "a scope(false) clears a set flag");
  }
  Check(Tp2TuningForThisThread(), "... and restores the previous value (true) on exit");
  SetTp2TuningForThisThread(false);

  {  // nesting: each level restores ITS predecessor, not the outermost value
    Tp2TuningScope a(true);
    {
      Tp2TuningScope b(false);
      Check(!Tp2TuningForThisThread(), "nested scope(false) inside scope(true)");
      {
        Tp2TuningScope c(true);
        Check(Tp2TuningForThisThread(), "third level");
      }
      Check(!Tp2TuningForThisThread(), "the third level restores the second's value");
    }
    Check(Tp2TuningForThisThread(), "the second level restores the first's value");
  }
  Check(!Tp2TuningForThisThread(), "the outermost scope restores the thread's original value");
  {  // a same-value scope changes nothing and restores nothing different
    SetTp2TuningForThisThread(true);
    {
      Tp2TuningScope s(true);
      Check(Tp2TuningForThisThread(), "scope(true) on a true thread");
    }
    Check(Tp2TuningForThisThread(), "... leaves it true");
    SetTp2TuningForThisThread(false);
  }
}

void Exceptions() {
  for (const bool start : {false, true}) {
    SetTp2TuningForThisThread(start);
    bool caught = false;
    try {
      Tp2TuningScope s(!start);
      Check(Tp2TuningForThisThread() == !start, "inside the throwing scope the flag is the scope's");
      throw std::runtime_error("a Model call that throws");
    } catch (const std::runtime_error&) {
      caught = true;
    }
    Check(caught, "the exception propagates out of the scope");
    Check(Tp2TuningForThisThread() == start, start ? "restored to true after a throw" : "restored to false after a throw");
  }
  // a throw from a NESTED scope unwinds both
  SetTp2TuningForThisThread(false);
  try {
    Tp2TuningScope a(true);
    Tp2TuningScope b(false);
    throw std::logic_error("deep");
  } catch (const std::logic_error&) {
  }
  Check(!Tp2TuningForThisThread(), "nested scopes both unwind on a throw");
  // a throwing guard inside a scope leaves the scope restored too
  SetTp2TuningForThisThread(false);
  try {
    Tp2TuningScope a(true);
    CheckTp2TuningScope(false, "guard");  // mismatches inside the scope
  } catch (const std::logic_error&) {
  }
  Check(!Tp2TuningForThisThread(), "the scope that held a failing guard restored the flag");
  SetTp2TuningForThisThread(false);
}

void ThreadIsolation() {
  SetTp2TuningForThisThread(false);
  bool other_default = true, other_inside = false;
  {
    Tp2TuningScope s(true);
    std::thread t([&] {
      other_default = Tp2TuningForThisThread();  // a fresh thread starts false, whatever this thread's scope says
      Tp2TuningScope inner(true);
      other_inside = Tp2TuningForThisThread();
    });
    t.join();
    Check(Tp2TuningForThisThread(), "another thread's scope does not touch this thread's flag");
  }
  Check(!other_default, "a new thread's flag is false even while another thread is inside a scope(true)");
  Check(other_inside, "a scope on another thread sets that thread's flag");
  // and the other direction: a flag set on a worker thread is invisible here
  bool worker_set = false;
  std::thread w([&] {
    SetTp2TuningForThisThread(true);
    worker_set = Tp2TuningForThisThread();
  });
  w.join();
  Check(worker_set && !Tp2TuningForThisThread(), "a flag set on a worker thread does not leak to this one");
}

void Guard() {
  SetTp2TuningForThisThread(false);
  CheckTp2TuningScope(false, "match false");
  SetTp2TuningForThisThread(true);
  CheckTp2TuningScope(true, "match true");
  bool threw = false;
  std::string msg;
  try {
    CheckTp2TuningScope(false, "Model::CheckTuningScope");
  } catch (const std::logic_error& e) {
    threw = true;
    msg = e.what();
  }
  Check(threw, "the guard throws std::logic_error when a TP=1 / stage Model runs on a thread flagged for a rank");
  Check(msg.find("Model::CheckTuningScope") != std::string::npos && msg.find("Tp2TuningScope") != std::string::npos &&
            msg.find("TP rank") != std::string::npos,
        "... and names the caller, the cure and the two layouts");
  SetTp2TuningForThisThread(false);
  threw = false;
  try {
    CheckTp2TuningScope(true, "x");
  } catch (const std::logic_error&) {
    threw = true;
  }
  Check(threw, "the guard throws for a rank Model on a thread flagged for TP=1 tunings");
  SetTp2TuningForThisThread(false);
}

// The hybrid rank worker: one thread, a rank Model and a stage Model, loaded in either order.
void TwoModelsOneThread() {
  for (const bool rank_first : {true, false}) {
    SetTp2TuningForThisThread(false);
    FakeModel first(rank_first), last(!rank_first);  // `first` is the rank Model when rank_first, the stage Model otherwise
    // NEGATIVE CONTROL: without scopes, Load's set-on-every-load leaves the flag at the LAST load, so exactly one of the two
    // Models is on the wrong tunings -- and the guard says so.
    Check(Throws(first) && !Throws(last),
          "without scopes the Model loaded FIRST fails the guard and the one loaded last passes (the flag follows the last load)");
    Check(Tp2TuningForThisThread() == last.tp2, "the flag follows the last load");
    // WITH scopes both are right, in any order, repeatedly, and the thread is left as it was found.
    const bool before = Tp2TuningForThisThread();
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
      try {
        first.ScopedCall();
        last.ScopedCall();
        last.ScopedCall();
        first.ScopedCall();
      } catch (...) {
        ok = false;
      }
    }
    Check(ok, "inside Tp2TuningScope(model's own value) both Models pass the guard, in any order");
    Check(Tp2TuningForThisThread() == before, "the scopes leave the thread's flag as they found it");
  }
  SetTp2TuningForThisThread(false);
}

// PickTuning really depends on the flag for a rank-equal shape: trellis 6144 x 5120 KB 5 M = 1 is gdn.in_proj_z at TP=1 and
// attn.qg of a TP rank, with different rows in gemm_tuning_table_trellis.inc / ..._tp2.inc. (If a retune makes the two rows
// equal this check has to move to another shape that still differs; the guard test above does not depend on it.)
void PickTuningFollowsTheScope() {
  SetTp2TuningForThisThread(false);
  const LinearTuning tp1 = PickTuning(Layout::kTrellis, 6144, 5120, 1, 5);
  LinearTuning tp2{};
  {
    Tp2TuningScope s(true);
    tp2 = PickTuning(Layout::kTrellis, 6144, 5120, 1, 5);
  }
  const LinearTuning after = PickTuning(Layout::kTrellis, 6144, 5120, 1, 5);
  Check(!SameTuning(tp1, tp2), "NEGATIVE CONTROL: a rank-equal shape picks a different row inside Tp2TuningScope(true)");
  Check(SameTuning(tp1, after), "after the scope the thread picks the TP=1 row again (the cache key carries the flag)");
  std::fprintf(stderr, "  trellis 6144x5120 KB5 M=1: TP=1 {WV %d SK %d MB %d NPW %d NT %d}  rank {WV %d SK %d MB %d NPW %d NT %d}\n", tp1.WV,
               tp1.SK, tp1.MB, tp1.NPW, tp1.NT, tp2.WV, tp2.SK, tp2.MB, tp2.NPW, tp2.NT);
  SetTp2TuningForThisThread(false);
}

}  // namespace

int main() {
  Raii();
  Exceptions();
  ThreadIsolation();
  Guard();
  TwoModelsOneThread();
  PickTuningFollowsTheScope();
  if (g_fails != 0) {
    std::fprintf(stderr, "test_tp_tuning_scope: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_tp_tuning_scope: PASS\n");
  return 0;
}

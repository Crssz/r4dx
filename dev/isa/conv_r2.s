	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	k_dflash_conv_t2_g16    ; -- Begin function k_dflash_conv_t2_g16
	.globl	k_dflash_conv_t2_g16
	.p2align	8
	.type	k_dflash_conv_t2_g16,@function
k_dflash_conv_t2_g16:                   ; @k_dflash_conv_t2_g16
; %bb.0:
	s_load_b128 s[12:15], s[0:1], 0x20
	v_lshl_add_u32 v0, ttmp9, 7, v0
	s_mov_b32 s2, exec_lo
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s15, v0
	s_cbranch_execz .LBB0_4
; %bb.1:
	s_load_b256 s[4:11], s[0:1], 0x0
	v_dual_mov_b32 v3, 0 :: v_dual_lshlrev_b32 v2, 3, v0
	v_bfe_u32 v14, v0, 1, 28
	s_mov_b32 s2, ttmp7
	s_ashr_i32 s3, ttmp7, 31
	s_ashr_i32 s17, s12, 31
	v_lshlrev_b64_e32 v[0:1], 1, v[2:3]
	s_ashr_i32 s19, s13, 31
	s_mov_b32 s18, s13
	s_mov_b32 s16, s12
	s_mul_u64 s[12:13], s[18:19], s[2:3]
	s_mul_u64 s[2:3], s[16:17], s[2:3]
	v_lshlrev_b32_e32 v2, 1, v14
	s_lshl_b64 s[12:13], s[12:13], 1
	s_lshl_b64 s[2:3], s[2:3], 1
	s_load_b32 s0, s[0:1], 0x30
	s_wait_kmcnt 0x0
	v_add_co_u32 v4, vcc_lo, s8, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s9, v1, vcc_lo
	s_add_nc_u64 s[6:7], s[6:7], s[12:13]
	s_add_nc_u64 s[4:5], s[4:5], s[2:3]
	global_load_u16 v2, v2, s[6:7]
	global_load_b128 v[6:9], v[4:5], off
	v_add_co_u32 v4, vcc_lo, s4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s5, v1, vcc_lo
	global_load_b128 v[10:13], v[4:5], off
	s_and_b32 s0, s0, ttmp7
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s0, 1
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v20, 16, v11
	v_and_b32_e32 v11, 0xffff0000, v11
	v_lshlrev_b32_e32 v2, 16, v2
	v_lshlrev_b32_e32 v17, 16, v8
	v_lshlrev_b32_e32 v16, 16, v7
	v_and_b32_e32 v7, 0xffff0000, v7
	v_and_b32_e32 v8, 0xffff0000, v8
	v_and_b32_e32 v22, 0xffff0000, v12
	v_and_b32_e32 v24, 0xffff0000, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v7, v2, v7 :: v_dual_lshlrev_b32 v18, 16, v9
	v_mul_f32_e32 v7, v7, v11
	v_dual_add_f32 v8, v2, v8 :: v_dual_lshlrev_b32 v15, 16, v6
	v_and_b32_e32 v9, 0xffff0000, v9
	v_lshlrev_b32_e32 v19, 16, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_add_f32 v12, v2, v15 :: v_dual_lshlrev_b32 v21, 16, v12
	v_dual_add_f32 v15, v2, v17 :: v_dual_and_b32 v6, 0xffff0000, v6
	v_dual_mul_f32 v8, v8, v22 :: v_dual_lshlrev_b32 v23, 16, v13
	v_add_f32_e32 v13, v2, v16
	v_add_f32_e32 v16, v2, v18
	v_add_f32_e32 v6, v2, v6
	v_add_f32_e32 v2, v2, v9
	v_mul_f32_e32 v12, v12, v19
	v_dual_mul_f32 v9, v15, v21 :: v_dual_and_b32 v10, 0xffff0000, v10
	v_mul_f32_e32 v13, v13, v20
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v11, v2, v24
	v_mul_f32_e32 v6, v6, v10
	v_mul_f32_e32 v10, v16, v23
	s_cbranch_scc1 .LBB0_3
; %bb.2:
	v_add_nc_u32_e32 v2, s14, v14
	s_lshl_b64 s[0:1], s[16:17], 1
	s_wait_alu depctr_sa_sdst(0)
	v_sub_co_u32 v4, vcc_lo, v4, s0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b64_e32 v[14:15], 1, v[2:3]
	s_wait_alu depctr_va_vcc(0)
	v_subrev_co_ci_u32_e64 v5, null, s1, v5, vcc_lo
	s_add_nc_u64 s[0:1], s[8:9], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v16, vcc_lo, s0, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s1, v1, vcc_lo
	v_add_co_u32 v18, vcc_lo, s6, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s7, v15, vcc_lo
	global_load_b128 v[14:17], v[16:17], off
	global_load_u16 v18, v[18:19], off
	global_load_b128 v[2:5], v[4:5], off
	s_wait_loadcnt 0x2
	v_lshlrev_b32_e32 v25, 16, v16
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v18, 16, v18
	v_lshlrev_b32_e32 v26, 16, v17
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v20, 16, v3
	v_lshlrev_b32_e32 v21, 16, v4
	v_dual_add_f32 v25, v18, v25 :: v_dual_and_b32 v16, 0xffff0000, v16
	v_add_f32_e32 v26, v18, v26
	v_lshlrev_b32_e32 v24, 16, v15
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_add_f32 v16, v18, v16 :: v_dual_and_b32 v15, 0xffff0000, v15
	v_fmac_f32_e32 v9, v25, v21
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_add_f32 v24, v18, v24 :: v_dual_lshlrev_b32 v19, 16, v2
	v_lshlrev_b32_e32 v22, 16, v5
	v_and_b32_e32 v5, 0xffff0000, v5
	v_fmac_f32_e32 v13, v24, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v10, v26, v22 :: v_dual_and_b32 v17, 0xffff0000, v17
	v_dual_add_f32 v17, v18, v17 :: v_dual_and_b32 v4, 0xffff0000, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v8, v16, v4 :: v_dual_lshlrev_b32 v23, 16, v14
	v_dual_fmac_f32 v11, v17, v5 :: v_dual_and_b32 v2, 0xffff0000, v2
	v_dual_add_f32 v15, v18, v15 :: v_dual_and_b32 v14, 0xffff0000, v14
	v_and_b32_e32 v3, 0xffff0000, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v23, v18, v23
	v_dual_add_f32 v14, v18, v14 :: v_dual_fmac_f32 v7, v15, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v12, v23, v19
	v_fmac_f32_e32 v6, v14, v2
.LBB0_3:
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_bfe_u32 v2, v12, 16, 1
	v_bfe_u32 v3, v13, 16, 1
	v_bfe_u32 v4, v6, 16, 1
	v_bfe_u32 v5, v9, 16, 1
	v_bfe_u32 v14, v8, 16, 1
	v_add3_u32 v2, v12, v2, 0x7fff
	v_add3_u32 v3, v13, v3, 0x7fff
	v_bfe_u32 v12, v10, 16, 1
	v_bfe_u32 v13, v11, 16, 1
	v_bfe_u32 v15, v7, 16, 1
	v_add3_u32 v9, v9, v5, 0x7fff
	v_add3_u32 v8, v8, v14, 0x7fff
	v_add3_u32 v5, v10, v12, 0x7fff
	v_add3_u32 v10, v11, v13, 0x7fff
	v_add3_u32 v7, v7, v15, 0x7fff
	v_add3_u32 v6, v6, v4, 0x7fff
	s_add_nc_u64 s[0:1], s[10:11], s[2:3]
	v_perm_b32 v4, v8, v9, 0x7060302
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_perm_b32 v5, v10, v5, 0x7060302
	v_perm_b32 v3, v7, v3, 0x7060302
	v_perm_b32 v2, v6, v2, 0x7060302
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	global_store_b128 v[0:1], v[2:5], off
.LBB0_4:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_dflash_conv_t2_g16
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 52
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 27
		.amdhsa_next_free_sgpr 20
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 8
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	k_dflash_conv_t2_g16, .Lfunc_end0-k_dflash_conv_t2_g16
                                        ; -- End function
	.set k_dflash_conv_t2_g16.num_vgpr, 27
	.set k_dflash_conv_t2_g16.num_agpr, 0
	.set k_dflash_conv_t2_g16.numbered_sgpr, 20
	.set k_dflash_conv_t2_g16.num_named_barrier, 0
	.set k_dflash_conv_t2_g16.private_seg_size, 0
	.set k_dflash_conv_t2_g16.uses_vcc, 1
	.set k_dflash_conv_t2_g16.uses_flat_scratch, 0
	.set k_dflash_conv_t2_g16.has_dyn_sized_stack, 0
	.set k_dflash_conv_t2_g16.has_recursion, 0
	.set k_dflash_conv_t2_g16.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 984
; TotalNumSgprs: 22
; NumVgprs: 27
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 27
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.ident	"clang version 22.1.8"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  generic
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 52
    .max_flat_workgroup_size: 1024
    .name:           k_dflash_conv_t2_g16
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         k_dflash_conv_t2_g16.kd
    .uses_dynamic_stack: false
    .vgpr_count:     27
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

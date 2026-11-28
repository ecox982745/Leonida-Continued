; Six sparse scalar inputs leave the first available three-location block at 9.
	v_interp_mov_f32_e32 v3, p0, attr0.x
	v_interp_mov_f32_e32 v4, p0, attr1.x
	v_add_f32_e32 v3, v3, v4
	v_interp_mov_f32_e32 v4, p0, attr2.x
	v_add_f32_e32 v3, v3, v4
	v_interp_mov_f32_e32 v4, p0, attr4.x
	v_add_f32_e32 v3, v3, v4
	v_interp_mov_f32_e32 v4, p0, attr6.x
	v_add_f32_e32 v3, v3, v4
	v_interp_mov_f32_e32 v4, p0, attr8.x
	v_add_f32_e32 v3, v3, v4
	v_mov_b32_e32 v0, 0.25
	v_mov_b32_e32 v1, 0.5
	v_interp_p1_f32_e32 v5, v0, attr20.x
	v_interp_p2_f32_e32 v5, v1, attr20.x
	v_interp_p1_f32_e32 v6, v0, attr21.x
	v_interp_p2_f32_e32 v6, v1, attr21.x
	exp mrt0 v3, v5, v6, v4 done vm
	s_endpgm

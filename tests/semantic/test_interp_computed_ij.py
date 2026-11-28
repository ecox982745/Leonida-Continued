"""SHADER-177: v_interp_p1/p2 through a barycentric pair the program computed.

Hardware weights the interpolation with the values the operand VGPRs hold at the two ops, not with the
hardware's own pair: Vonsek measured it on RDNA2 (#1224, synthetic 64x64 triangle, attr (0,0)/(2,0)/(0,2))
-- zeroing v0/v1 before the interp renders (0,0), i.e. P0, on every pixel, and reading the pair through
swapped copies renders (v,u). The single-lane model used to ignore the operand, so those translated as the
centre interpolation. Now the pair's attribute is read per-vertex and weighted with the operand values:

* a v_mov of a constant into v0/v1 (snapped coordinates, the zero form of the evidence above);
* arithmetic on the saved pair (evaluate-at-offset: J' = J / 2 halves the second weight);
* copies in either role, swapped included -- as writes into the operand registers; a pristine plain
  copy of the pair stays #1224's model (centre interpolation, bit-identical .spv);
* the corpus form where J is written between the p1 and the p2: the p1's I is kept and the p2's J is
  loaded fresh, because hardware's final value uses the I of the p1 with the J of the p2;
* v_interp_mov_f32 still reads the provoking vertex when the location also has computed p1/p2.

Refused instead of guessed (Unsupported, naming the pattern): a source that is neither v0/v1 nor
written, or a pristine copy in the wrong role, is the #1224 guard's refusal (interp_pair_src, at
emission); here: a computed pair whose p1 partial is rewritten or read between the ops (its hardware value P10*I is then
observable), one whose ops are not in a single basic block (the captured I would not dominate the p2), a
dangling op without its partner, and any computed pair under --f16-interp-fallback alone, which has no
per-vertex inputs. Under --bary-fallback (also with --f16-interp-fallback) the attribute is a Flat vec4[3] that the
pass-through geometry shader fills with the three vertices' values: ComputedIJBaryFallback. A pristine pair (v0/v1 nothing writes) keeps the interpolated-input read: no PerVertexKHR.

Pipodi's straight-line scan over the corpus found the affected 213 PS (upper bound; issue #1218,
comment 7076614458); the classification and counts are in docs/design.md's SHADER-177 row.
"""
import os
import shutil
import struct
import subprocess
import tempfile
import unittest

import support
import rdna2spirv

# Barycentric weights of vertices 0, 1, 2 as Vulkan hands them over ((k, i, j); AMD's i, j are .y/.z).
BARY = (0.25, 0.25, 0.5)
I, J = BARY[1], BARY[2]
# Vonsek's triangle: attr0.x = u = 0/2/0, attr0.y = v = 0/0/2, plus channels for the mov and snap checks.
UV = ((0.0, 0.0, 1.0, 0.0), (2.0, 0.0, 3.0, 0.0), (0.0, 2.0, 5.0, 0.0))
ATTR0 = [tuple(struct.unpack("<I", struct.pack("<f", x))[0] for x in v) for v in UV]


def run(asm, attrs=None):
    spv = support.translate_lines("ps_ij", asm)
    return spv, support.mrt0_words(spv, inputs=support.varyings(spv, attrs or {0: ATTR0}, BARY))


def floats(words):
    return struct.unpack("<" + "f" * len(words), struct.pack("<" + "I" * len(words), *words))


class ComputedIJ(unittest.TestCase):
    def test_swap_reads_both_incoming_barycentrics(self) -> None:
        _, words = run(["v_swap_b32 v0, v1", "exp mrt0 v0, v1, v0, v1 done vm", "s_endpgm"])
        self.assertEqual(floats(words), (J, I, J, I))

    def test_swap_overwrites_interpolation_weight_in_second_operand(self) -> None:
        # A copy of I ceases to be pristine too when it is the second operand of the swap.
        for weight in ("v0", "v8"):
            with self.subTest(weight=weight):
                setup = ["v_mov_b32 v8, v0"] if weight == "v8" else []
                _, words = run(setup + [
                    "v_mov_b32 v6, 0.125", f"v_swap_b32 v6, {weight}",
                    f"v_interp_p1_f32 v2, {weight}, attr0.x", "v_interp_p2_f32 v2, v1, attr0.x",
                    f"v_interp_p1_f32 v3, {weight}, attr0.y", "v_interp_p2_f32 v3, v1, attr0.y",
                    "exp mrt0 v2, v3, v6, v2 done vm", "s_endpgm",
                ])
                self.assertEqual(floats(words), (0.25, 2 * J, I, 0.25))

    def test_zeroed_pair_reads_the_provoking_vertex(self):
        """Vonsek's zero-source row: I' = J' = 0 interpolate to P0 = (0, 0) on every pixel."""
        _, w = run(["v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0",
                    "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                    "v_interp_p1_f32_e32 v3, v0, attr0.y", "v_interp_p2_f32_e32 v3, v1, attr0.y",
                    "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"])
        self.assertEqual(floats(w[:2]), (0.0, 0.0))

    def test_constant_pair_snaps_to_a_point(self):
        """Snapped coordinates: I' = J' = 0.25 pick the point a quarter along both edges."""
        _, w = run(["v_mov_b32_e32 v0, 0.25", "v_mov_b32_e32 v1, 0.25",
                    "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                    "v_interp_p1_f32_e32 v3, v0, attr0.y", "v_interp_p2_f32_e32 v3, v1, attr0.y",
                    "v_interp_p1_f32_e32 v4, v0, attr0.z", "v_interp_p2_f32_e32 v4, v1, attr0.z",
                    "exp mrt0 v2, v3, v4, v2 done vm", "s_endpgm"])
        # P = 0.5*P0 + 0.25*P1 + 0.25*P2
        self.assertEqual(floats(w[:3]), (0.5, 0.5, 2.5))

    def test_arithmetic_pair_evaluates_at_an_offset(self):
        """Evaluate-at-offset: keeping I and halving J halves the second weight (u stays, v halves)."""
        _, w = run(["v_mov_b32_e32 v10, v0", "v_mul_f32_e32 v11, 0.5, v1",
                    "v_mov_b32_e32 v0, v10", "v_mov_b32_e32 v1, v11",
                    "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                    "v_interp_p1_f32_e32 v3, v0, attr0.y", "v_interp_p2_f32_e32 v3, v1, attr0.y",
                    "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"])
        self.assertEqual(floats(w[:2]), (2 * I, J))

    def test_swapped_copies_swap_the_weights(self):
        """Vonsek's role-swap row as a computed pair: the saved J is written into v0 and the saved I
        into v1, so the p1 weights with J and the p2 with I: (v, u). Pristine copies read in swapped
        roles carry no computed value and stay the #1224 guard's refusal (root-role enforced)."""
        _, w = run(["v_mov_b32_e32 v14, v1", "v_mov_b32_e32 v15, v0",    # save J, save I
                    "v_mov_b32_e32 v0, v14", "v_mov_b32_e32 v1, v15",    # swap into the operands
                    "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                    "v_interp_p1_f32_e32 v3, v0, attr0.y", "v_interp_p2_f32_e32 v3, v1, attr0.y",
                    "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"])
        self.assertEqual(floats(w[:2]), (2 * J, 2 * I))

    def test_j_written_between_the_ops(self):
        """The corpus form: the p1 runs with I already computed, and J is set up between the two ops."""
        _, w = run(["v_mov_b32_e32 v16, 1.0", "v_mov_b32_e32 v0, v15",     # I' = 0 (v15 never written)
                    "v_interp_p1_f32_e32 v2, v0, attr0.x",
                    "v_mul_f32_e32 v1, 0.25, v16",                          # J' written after the p1
                    "v_interp_p2_f32_e32 v2, v1, attr0.x",
                    "v_interp_p1_f32_e32 v3, v0, attr0.y",
                    "v_mul_f32_e32 v1, 0.5, v16",
                    "v_interp_p2_f32_e32 v3, v1, attr0.y",
                    "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"])
        # x: 0 + 0*2 + 0.25*0 = 0; y: 0 + 0*0 + 0.5*2 = 1
        self.assertEqual(floats(w[:2]), (0.0, 1.0))

    def test_mov_reads_the_provoking_vertex_of_a_computed_location(self):
        """v_interp_mov_f32 of a location the computed pair also reads still gives vertex 0's value
        (1.0; the perspective interpolation of 1/3/5 at these weights is 3.5)."""
        _, w = run(["v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0",
                    "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                    "v_interp_mov_f32_e32 v3, p0, attr0.z",
                    "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"])
        self.assertEqual(floats(w[:2]), (0.0, 1.0))

    def test_f16_pair_through_a_computed_pair(self):
        """The f16 forms weight with the computed pair too, both halves."""
        LO, HI = (0.5, 1.5, 2.5), (-4.0, 8.0, 16.0)
        pack = lambda lo, hi: struct.unpack("<I", struct.pack("<ee", lo, hi))[0]
        attr0 = [(pack(lo, hi), 0, 0, 0) for lo, hi in zip(LO, HI)]
        _, w = run(["v_mov_b32_e32 v0, 0.5", "v_mov_b32_e32 v1, 0",
                    "v_interp_p1ll_f16 v2, v0, attr0.x", "v_interp_p2_f16 v2, v1, attr0.x, v2",
                    "v_interp_p1ll_f16 v3, v0, attr0.x high", "v_interp_p2_f16 v3, v1, attr0.x, v3 high",
                    "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"], attrs={0: attr0})
        # (k, i, j) = (0.5, 0.5, 0): low 0.5*0.5 + 0.5*1.5 = 1.0; high 0.5*-4 + 0.5*8 = 2.0
        self.assertEqual(f"{w[0] & 0xFFFF:#06x}", "0x3c00")
        self.assertEqual(f"{w[1] & 0xFFFF:#06x}", "0x4000")

    def test_pristine_pair_still_reads_the_interpolated_input(self):
        """A pair nothing writes keeps the single-lane model's own read: no PerVertexKHR, no %bary."""
        spv, w = run(["v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                      "v_interp_p1_f32_e32 v3, v0, attr0.y", "v_interp_p2_f32_e32 v3, v1, attr0.y",
                      "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"])
        self.assertEqual(floats(w[:2]), (2 * I, 2 * J))
        self.assertNotIn("PerVertexKHR", spv)
        self.assertNotIn("%bary", spv)

    def test_only_i_overwritten_preserves_hardware_j(self):
        """Overwriting only v0 promotes the pair to computed I/J; untouched v1 must still load BaryCoordKHR.z."""
        _, w = run(["v_mov_b32_e32 v0, 0",
                    "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                    "v_interp_p1_f32_e32 v3, v0, attr0.y", "v_interp_p2_f32_e32 v3, v1, attr0.y",
                    "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"])
        self.assertEqual(floats(w[:2]), (0.0, 2 * J))

    def test_only_j_overwritten_preserves_hardware_i(self):
        """Overwriting only v1 promotes the pair to computed I/J; untouched v0 must still load BaryCoordKHR.y."""
        _, w = run(["v_mov_b32_e32 v1, 0",
                    "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                    "v_interp_p1_f32_e32 v3, v0, attr0.y", "v_interp_p2_f32_e32 v3, v1, attr0.y",
                    "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"])
        self.assertEqual(floats(w[:2]), (2 * I, 0.0))

    def test_conditional_overwrite_preserves_hardware_barycentrics_on_untaken_path(self):
        """When a branch conditionally overwrites v0, the untaken path must still interpolate with hardware (I, J)."""
        asm = [
            "s_cmp_eq_u32 s0, 1    ; encoding: [0,0,0,0]",
            "s_cbranch_scc0 1    ; encoding: [0,0,0,0]",
            "v_mov_b32_e32 v0, 0    ; encoding: [0,0,0,0]",
            "v_interp_p1_f32_e32 v2, v0, attr0.x    ; encoding: [0,0,0,0]",
            "v_interp_p2_f32_e32 v2, v1, attr0.x    ; encoding: [0,0,0,0]",
            "v_interp_p1_f32_e32 v3, v0, attr0.y    ; encoding: [0,0,0,0]",
            "v_interp_p2_f32_e32 v3, v1, attr0.y    ; encoding: [0,0,0,0]",
            "exp mrt0 v2, v3, v2, v3 done vm    ; encoding: [0,0,0,0,0,0,0,0]",
            "s_endpgm    ; encoding: [0,0,0,0]",
        ]
        spv = support.translate_lines("ps_ij_cond", asm)
        inputs = support.varyings(spv, {0: ATTR0}, BARY)
        w_untaken = support.mrt0_words(spv, inputs=inputs, push_constants=[0] * 36)
        w_taken = support.mrt0_words(spv, inputs=inputs, push_constants=[1] + [0] * 35)
        self.assertEqual(floats(w_untaken[:2]), (2 * I, 2 * J))
        self.assertEqual(floats(w_taken[:2]), (0.0, 2 * J))

    def test_exec_masked_overwrite_preserves_hardware_barycentrics_in_masked_lanes(self):
        """EXEC can mask a VGPR write in straight-line code without a branch; masked lanes still hold hardware I/J."""
        asm = [
            "s_mov_b32 exec_lo, s0",
            "v_mov_b32_e32 v0, 0",
            "s_mov_b32 exec_lo, -1",
            "v_interp_p1_f32_e32 v2, v0, attr0.x",
            "v_interp_p2_f32_e32 v2, v1, attr0.x",
            "v_interp_p1_f32_e32 v3, v0, attr0.y",
            "v_interp_p2_f32_e32 v3, v1, attr0.y",
            "exp mrt0 v2, v3, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ij_exec", asm)
        inputs = support.varyings(spv, {0: ATTR0}, BARY)
        w_masked = support.mrt0_words(spv, inputs=inputs, push_constants=[0] * 36)
        w_active = support.mrt0_words(spv, inputs=inputs, push_constants=[0xFFFFFFFF] + [0] * 35)
        self.assertEqual(floats(w_masked[:2]), (2 * I, 2 * J))
        self.assertEqual(floats(w_active[:2]), (0.0, 2 * J))

        # Also check saveexec form loads both barycentric components in the prologue.
        spv_save = support.translate_lines("ps_ij_saveexec", [
            "s_and_saveexec_b32 s10, s0",
            "v_mov_b32_e32 v0, 0",
            "s_mov_b32 exec_lo, s10",
            "v_interp_p1_f32_e32 v2, v0, attr0.x",
            "v_interp_p2_f32_e32 v2, v1, attr0.x",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ])
        self.assertIn("%bc1 = OpCompositeExtract %float %bc 1", spv_save)
        self.assertIn("%bc2 = OpCompositeExtract %float %bc 2", spv_save)

    def test_conditional_overwrite_then_copied_coords_preserves_hardware_barycentrics(self):
        """After a conditional overwrite of v0, copying v0/v1 into temporaries for v_interp must still see hardware I/J."""
        asm = [
            "s_cmp_eq_u32 s0, 1    ; encoding: [0,0,0,0]",
            "s_cbranch_scc0 1    ; encoding: [0,0,0,0]",
            "v_mov_b32_e32 v0, 0    ; encoding: [0,0,0,0]",
            "v_mov_b32_e32 v5, v0    ; encoding: [0,0,0,0]",
            "v_mov_b32_e32 v6, v1    ; encoding: [0,0,0,0]",
            "v_interp_p1_f32_e32 v2, v5, attr0.x    ; encoding: [0,0,0,0]",
            "v_interp_p2_f32_e32 v2, v6, attr0.x    ; encoding: [0,0,0,0]",
            "v_interp_p1_f32_e32 v3, v5, attr0.y    ; encoding: [0,0,0,0]",
            "v_interp_p2_f32_e32 v3, v6, attr0.y    ; encoding: [0,0,0,0]",
            "exp mrt0 v2, v3, v2, v3 done vm    ; encoding: [0,0,0,0,0,0,0,0]",
            "s_endpgm    ; encoding: [0,0,0,0]",
        ]
        spv = support.translate_lines("ps_ij_cond_copy", asm)
        inputs = support.varyings(spv, {0: ATTR0}, BARY)
        w_untaken = support.mrt0_words(spv, inputs=inputs, push_constants=[0] * 36)
        w_taken = support.mrt0_words(spv, inputs=inputs, push_constants=[1] + [0] * 35)
        self.assertEqual(floats(w_untaken[:2]), (2 * I, 2 * J))
        self.assertEqual(floats(w_taken[:2]), (0.0, 2 * J))


class ComputedIJRefusals(unittest.TestCase):
    def refuse(self, asm, message):
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            support.translate_lines("ps_ij_refuse", asm)
        self.assertIn(message, str(cm.exception))

    def test_unwritten_source_is_not_a_pair(self):
        # delegated to the #1224 guard (interp_pair_src), whose refusal names the role
        self.refuse(["v_interp_p1_f32_e32 v2, v9, attr0.x", "v_interp_p2_f32_e32 v2, v9, attr0.x",
                     "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"],
                    "v_interp_p1_f32 I source v9: only v0 (or a plain v_mov copy of it) is modelled")

    def test_dangling_p1(self):
        self.refuse(["v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0",
                     "v_interp_p1_f32_e32 v2, v0, attr0.x",
                     "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"],
                    "v_interp_p1_f32 without its p2 partner")

    def test_partial_rewritten_between_the_ops(self):
        self.refuse(["v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0",
                     "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_mov_b32_e32 v2, 1.0",
                     "v_interp_p2_f32_e32 v2, v1, attr0.x",
                     "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"],
                    "the p1 partial is rewritten between the ops")

    def test_partial_read_between_the_ops(self):
        self.refuse(["v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0",
                     "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_add_f32_e32 v5, v2, 1.0",
                     "v_interp_p2_f32_e32 v2, v1, attr0.x",
                     "exp mrt0 v2, v5, v2, v2 done vm", "s_endpgm"],
                    "the p1 partial is read between the ops")

    def test_pair_split_across_blocks(self):
        # Encodings make the program counters advance so the branch lands the p2 in its own block
        # (the s_branch at pc 16 skips the three nops to pc 28).
        lines = ["v_mov_b32_e32 v0, 0    ; encoding: [0,0,0,0]",
                 "v_mov_b32_e32 v1, 0    ; encoding: [0,0,0,0]",
                 "v_interp_p1_f32_e32 v2, v0, attr0.x    ; encoding: [0,0,0,0]",
                 "s_branch 3    ; encoding: [0,0,0,0]",
                 "nop    ; encoding: [0,0,0,0]",
                 "nop    ; encoding: [0,0,0,0]",
                 "nop    ; encoding: [0,0,0,0]",
                 "v_interp_p2_f32_e32 v2, v1, attr0.x    ; encoding: [0,0,0,0]",
                 "exp mrt0 v2, v2, v2, v2 done vm",
                 "s_endpgm"]
        self.refuse(lines, "the pair's ops are not in one basic block")

    def test_computed_pair_under_f16_interp_fallback(self):
        # --f16-interp-fallback reads decoded halves from the producer: no per-vertex inputs to
        # weight with a computed pair (--bary-fallback gets them from its geometry shader: ComputedIJBaryFallback)
        old = rdna2spirv.F16_HALVES
        rdna2spirv.F16_HALVES = True
        try:
            self.refuse(["v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0",
                         "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                         "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"],
                        "computed I/J pair under --f16-interp-fallback")
        finally:
            rdna2spirv.F16_HALVES = old


# The ComputedIJ programs, for the --bary-fallback comparison.
_ZERO = ["v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0",
         "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
         "v_interp_p1_f32_e32 v3, v0, attr0.y", "v_interp_p2_f32_e32 v3, v1, attr0.y",
         "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"]
_PROGRAMS = {
    "zeroed": _ZERO,
    "snapped": ["v_mov_b32_e32 v0, 0.25", "v_mov_b32_e32 v1, 0.25",
                "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                "v_interp_p1_f32_e32 v3, v0, attr0.y", "v_interp_p2_f32_e32 v3, v1, attr0.y",
                "v_interp_p1_f32_e32 v4, v0, attr0.z", "v_interp_p2_f32_e32 v4, v1, attr0.z",
                "exp mrt0 v2, v3, v4, v2 done vm", "s_endpgm"],
    "offset": ["v_mov_b32_e32 v10, v0", "v_mul_f32_e32 v11, 0.5, v1",
               "v_mov_b32_e32 v0, v10", "v_mov_b32_e32 v1, v11",
               "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
               "v_interp_p1_f32_e32 v3, v0, attr0.y", "v_interp_p2_f32_e32 v3, v1, attr0.y",
               "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"],
    "swapped": ["v_mov_b32_e32 v14, v1", "v_mov_b32_e32 v15, v0",
                "v_mov_b32_e32 v0, v14", "v_mov_b32_e32 v1, v15",
                "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                "v_interp_p1_f32_e32 v3, v0, attr0.y", "v_interp_p2_f32_e32 v3, v1, attr0.y",
                "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"],
    "j_between": ["v_mov_b32_e32 v16, 1.0", "v_mov_b32_e32 v0, v15",
                  "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_mul_f32_e32 v1, 0.25, v16",
                  "v_interp_p2_f32_e32 v2, v1, attr0.x", "v_interp_p1_f32_e32 v3, v0, attr0.y",
                  "v_mul_f32_e32 v1, 0.5, v16", "v_interp_p2_f32_e32 v3, v1, attr0.y",
                  "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"],
    "mov": ["v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0",
            "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
            "v_interp_mov_f32_e32 v3, p0, attr0.z",
            "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"],
    "mov_p10_p20": ["v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0",
                    "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                    "v_interp_mov_f32_e32 v3, p10, attr0.y", "v_interp_mov_f32_e32 v4, p20, attr0.z",
                    "exp mrt0 v2, v3, v4, v3 done vm", "s_endpgm"],
}


class ComputedIJBaryFallback(unittest.TestCase):
    """--bary-fallback: a computed pair's attribute is a Flat vec4[3] its geometry shader fills with the three vertices'
    values (tools/bary_fallback_gs.py, "per_vertex" in bary_fallback_input), weighted as on the PerVertexKHR path."""

    def translate(self, asm, bary=True, f16=False):
        old = rdna2spirv.BARY_FALLBACK, rdna2spirv.F16_HALVES
        rdna2spirv.BARY_FALLBACK, rdna2spirv.F16_HALVES = bary, f16
        try:
            spv = support.translate_lines("ps_ij_fb", asm)
            meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
        finally:
            rdna2spirv.BARY_FALLBACK, rdna2spirv.F16_HALVES = old
        return spv, meta

    def words(self, spv, attrs=None):
        return support.mrt0_words(spv, inputs=support.varyings(spv, attrs or {0: ATTR0}, BARY))

    def test_same_values_as_the_per_vertex_path(self):
        for name, asm in _PROGRAMS.items():
            with self.subTest(program=name):
                native, _ = self.translate(asm, bary=False)
                fb, _ = self.translate(asm)
                self.assertEqual(self.words(fb), self.words(native))

    def test_module_has_no_per_vertex_khr(self):
        spv, meta = self.translate(_ZERO)
        self.assertNotIn("PerVertexKHR", spv)
        self.assertNotIn("FragmentBarycentricKHR", spv)
        self.assertNotIn("SPV_KHR_fragment_shader_barycentric", spv)
        self.assertIn("%in_attr0 = OpVariable %_ptr_Input_pv4 Input", spv)
        self.assertIn("OpDecorate %in_attr0 Flat", spv)
        self.assertIn("OpDecorate %in_attr0 Location 0", spv)
        fb = meta["bary_fallback_input"]
        self.assertEqual(fb["per_vertex"], {"0": 0})
        self.assertNotIn(fb["location"], (0, 1, 2))
        self.assertEqual(meta["inputs"], [0])        # the producer's attribute, for the geometry shader
        self.assertNotIn("fragmentShaderBarycentric", meta["features"])

    def test_the_array_avoids_other_inputs(self):
        """attr1 read at location 1: attr0's three locations go past it (2..4), %bary to the free location 0."""
        spv, meta = self.translate(["v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0",
                                    "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
                                    "v_interp_mov_f32_e32 v3, p0, attr1.x",
                                    "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"])
        fb = meta["bary_fallback_input"]
        self.assertEqual(fb["per_vertex"], {"0": 2})
        self.assertEqual(fb["location"], 0)
        self.assertIn("OpDecorate %in_attr0 Location 2", spv)
        self.assertIn("OpDecorate %in_attr1 Location 1", spv)
        self.assertEqual(meta["inputs"], [0, 1])

    def test_sparse_inputs_leave_room_for_two_per_vertex_arrays(self):
        path = os.path.join(support.HERE, "fixtures", "sparse_bary_locations.asm")
        with open(path, encoding="utf-8") as fixture:
            spv, meta = self.translate(fixture.read().splitlines())
        self.assertEqual(meta["bary_fallback_input"]["per_vertex"], {"20": 9, "21": 12})
        self.assertEqual(meta["bary_fallback_input"]["location"], 3)
        self.assertEqual(meta["inputs"], [0, 1, 2, 4, 6, 8, 20, 21])
        self.assertIn("OpDecorate %in_attr20 Location 9", spv)
        self.assertIn("OpDecorate %in_attr21 Location 12", spv)

    def test_f16_pair_with_the_f16_fallback_too(self):
        """NVIDIA Pascal takes both fallbacks: a computed f16 pair decodes the three vertices' raw words rather than
        the producer's decoded halves, and gives the PerVertexKHR path's halves."""
        lo_hi = ((0.5, -4.0), (1.5, 8.0), (2.5, 16.0))
        attrs = {0: [(struct.unpack("<I", struct.pack("<ee", lo, hi))[0], 0, 0, 0) for lo, hi in lo_hi]}
        asm = ["v_mov_b32_e32 v0, 0.5", "v_mov_b32_e32 v1, 0",
               "v_interp_p1ll_f16 v2, v0, attr0.x", "v_interp_p2_f16 v2, v1, attr0.x, v2",
               "v_interp_p1ll_f16 v3, v0, attr0.x high", "v_interp_p2_f16 v3, v1, attr0.x, v3 high",
               "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"]
        native, _ = self.translate(asm, bary=False)
        fb, meta = self.translate(asm, f16=True)
        self.assertNotIn("%in_f16h0", fb)
        self.assertNotIn("PerVertexKHR", fb)
        self.assertEqual(meta["bary_fallback_input"]["per_vertex"], {"0": 0})
        w = self.words(fb, attrs)
        self.assertEqual(w, self.words(native, attrs))
        self.assertEqual(f"{w[0] & 0xFFFF:#06x}", "0x3c00")
        self.assertEqual(f"{w[1] & 0xFFFF:#06x}", "0x4000")

    def test_f16_fallback_alone_still_refuses(self):
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            self.translate(_ZERO, bary=False, f16=True)
        self.assertIn("computed I/J pair under --f16-interp-fallback", str(cm.exception))

    def test_geometry_shader_writes_the_three_vertices(self):
        import bary_fallback_gs
        _, meta = self.translate(_ZERO)
        fb = meta["bary_fallback_input"]
        gs = bary_fallback_gs.gs_spvasm(meta["inputs"], fb["location"], per_vertex=fb["per_vertex"])
        self.assertIn("OpDecorate %pvin0 Location 0", gs)
        self.assertIn("OpDecorate %pvout0 Location 0", gs)
        self.assertIn("OpDecorate %pvout0 Flat", gs)
        self.assertNotIn("%out0 ", gs)                 # not also passed through as an interpolated copy
        for v in range(3):                            # at every emitted vertex, element k = vertex k's value
            for k in range(3):
                self.assertIn(f"%p{v}_pvin0_{k} = OpAccessChain %ptr_in_v4 %pvin0 %int_{k}", gs)
                self.assertIn(f"%q{v}_pvout0_{k} = OpAccessChain %ptr_out_v4 %pvout0 %int_{k}", gs)
        with self.assertRaises(ValueError):           # the array may not overlap a pass-through or %bary
            bary_fallback_gs.gs_spvasm([0, 1], 4, per_vertex={0: 1})
        with self.assertRaises(ValueError):
            bary_fallback_gs.gs_spvasm([0], 2, per_vertex={0: 0})

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val")
    def test_geometry_shader_validates(self):
        """attr1 passed through, attr0's three vertices at 2..4, %bary at 0 (test_the_array_avoids_other_inputs)."""
        import bary_fallback_gs
        with tempfile.TemporaryDirectory() as tmp:
            asm, spv = os.path.join(tmp, "gs.spvasm"), os.path.join(tmp, "gs.spv")
            with open(asm, "w", encoding="utf-8") as f:
                f.write(bary_fallback_gs.gs_spvasm([0, 1], 0, per_vertex={0: 2}))
            for cmd in (["spirv-as", "--target-env", "vulkan1.2", asm, "-o", spv],
                        ["spirv-val", "--target-env", "vulkan1.2", spv]):
                r = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()

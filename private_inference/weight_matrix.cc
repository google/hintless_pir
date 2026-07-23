/*
 * Copyright 2026 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "private_inference/weight_matrix.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "private_inference/parameters.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/rns/rns_bfv_ciphertext.h"
#include "shell_encryption/status_macros.h"

namespace private_inference {

absl::StatusOr<std::unique_ptr<WeightMatrix>> WeightMatrix::Create(
    const RnsContext* rns_context, const RnsGadget* gadget,
    const absl::string_view prng_seed_ct_pad,
    const absl::string_view prng_seed_gk_pad) {
  if (rns_context == nullptr) {
    return absl::InvalidArgumentError("`rns_context` must not be null.");
  }

  std::vector<const PrimeModulus*> rns_moduli = rns_context->MainPrimeModuli();
  RLWE_ASSIGN_OR_RETURN(Encoder encoder, Encoder::Create(rns_context));

  std::string prng_seed_ct_pad_(prng_seed_ct_pad);
  std::string prng_seed_gk_pad_(prng_seed_gk_pad);
  if (prng_seed_ct_pad.empty()) {
    RLWE_ASSIGN_OR_RETURN(prng_seed_ct_pad_,
                          rlwe::SingleThreadHkdfPrng::GenerateSeed());
  }
  if (prng_seed_gk_pad.empty()) {
    RLWE_ASSIGN_OR_RETURN(prng_seed_gk_pad_,
                          rlwe::SingleThreadHkdfPrng::GenerateSeed());
  }

  return absl::WrapUnique(new WeightMatrix(
      rns_context, std::move(rns_moduli), gadget, std::move(encoder),
      std::move(prng_seed_ct_pad_), std::move(prng_seed_gk_pad_)));
}

absl::Status WeightMatrix::SetData(
    const std::vector<std::vector<Integer>>& data) {
  int num_rows = data.size();
  int num_cols = data[0].size();
  if (num_rows % 2 == 1 || num_cols % 2 == 1) {
    return absl::InvalidArgumentError(
        "`data` cannot have odd number of rows or columns,");
  }
  int num_slots_per_group = 1 << (rns_context_->LogN() - 1);
  int num_slots = num_slots_per_group * 2;
  if (num_cols > num_slots) {
    return absl::InvalidArgumentError(
        "number of columns cannot be more than 2 * RLWE dimension.");
  }
  if (num_rows > num_slots) {
    return absl::InvalidArgumentError(
        "number of rows cannot be more than 2 * RLWE dimension.");
  }

  // We divide a matrix into 2 x 2 blocks, each block is divided horizontally
  // into two and then padded to a matrix of dimension d x 2*d, for d = number
  // slots per group, as we have two cyclic subgroups among the slots.
  //     0    ..     k/2       .. d-1 |    d      ..             .. 2d-1
  // (B[0][0] .. B[0][k/2-1] 0 .. 0   | B[0][k/2] .. B[0][k-1] 0 .. 0   )
  // (...                             |                                 )
  int d = num_slots_per_group;
  diagonals00_.reserve(d);
  diagonals01_.reserve(d);
  diagonals10_.reserve(d);
  diagonals11_.reserve(d);

  BlockReader<Integer> reader00(num_rows / 2, num_cols / 2, 0, 0, d);
  for (int i = 0; i < d; ++i) {  // i'th diagonal
    std::vector<Integer> slot_values(num_slots, 0);

    for (int j = 0; j < d; ++j) {
      slot_values[j] = reader00.GetDiagonal(i, j, data);
      slot_values[d + j] = reader00.GetDiagonal(num_cols / 4 + i, j, data);
    }
    RLWE_ASSIGN_OR_RETURN(
        RnsPolynomial diagonal,
        encoder_.EncodeBfv(slot_values, rns_moduli_, /*is_scaled=*/false));
    diagonals00_.push_back(std::move(diagonal));
  }

  BlockReader<Integer> reader01(num_rows / 2, num_cols / 2, 0, num_cols / 2, d);
  for (int i = 0; i < d; ++i) {  // i'th diagonal
    std::vector<Integer> slot_values(num_slots, 0);

    for (int j = 0; j < d; ++j) {
      slot_values[j] = reader01.GetDiagonal(i, j, data);
      slot_values[d + j] = reader01.GetDiagonal(num_cols / 4 + i, j, data);
    }
    RLWE_ASSIGN_OR_RETURN(
        RnsPolynomial diagonal,
        encoder_.EncodeBfv(slot_values, rns_moduli_, /*is_scaled=*/false));
    diagonals01_.push_back(std::move(diagonal));
  }

  BlockReader<Integer> reader10(num_rows / 2, num_cols / 2, num_rows / 2, 0, d);
  for (int i = 0; i < d; ++i) {  // i'th diagonal
    std::vector<Integer> slot_values(num_slots, 0);

    for (int j = 0; j < d; ++j) {
      slot_values[j] = reader10.GetDiagonal(i, j, data);
      slot_values[d + j] = reader10.GetDiagonal(num_cols / 4 + i, j, data);
    }
    RLWE_ASSIGN_OR_RETURN(
        RnsPolynomial diagonal,
        encoder_.EncodeBfv(slot_values, rns_moduli_, /*is_scaled=*/false));
    diagonals10_.push_back(std::move(diagonal));
  }

  BlockReader<Integer> reader11(num_rows / 2, num_cols / 2, num_rows / 2,
                                num_cols / 2, d);
  for (int i = 0; i < d; ++i) {  // i'th diagonal
    std::vector<Integer> slot_values(num_slots, 0);

    for (int j = 0; j < d; ++j) {
      slot_values[j] = reader11.GetDiagonal(i, j, data);
      slot_values[d + j] = reader11.GetDiagonal(num_cols / 4 + i, j, data);
    }
    RLWE_ASSIGN_OR_RETURN(
        RnsPolynomial diagonal,
        encoder_.EncodeBfv(slot_values, rns_moduli_, /*is_scaled=*/false));
    diagonals11_.push_back(std::move(diagonal));
  }

  return absl::OkStatus();
}

absl::Status WeightMatrix::PreprocessPad(int num_rotations) {
  if (rns_gadget_ == nullptr) {
    return absl::InvalidArgumentError("rns_gadget_ is null");
  }

  int log_n = rns_context_->LogN();
  int gadget_dim = rns_gadget_->Dimension();
  RLWE_ASSIGN_OR_RETURN(auto prng_ct_pad,
                        rlwe::SingleThreadHkdfPrng::Create(prng_seed_ct_pad_));
  RLWE_ASSIGN_OR_RETURN(
      RnsPolynomial ct_pad,
      RnsPolynomial::SampleUniform(log_n, prng_ct_pad.get(), rns_moduli_));
  RLWE_RETURN_IF_ERROR(ct_pad.NegateInPlace(rns_moduli_));

  RLWE_ASSIGN_OR_RETURN(gk_pads_, RnsGaloisKey::SampleRandomPad(
                                      gadget_dim, log_n, rns_moduli_,
                                      prng_seed_gk_pad_, rlwe::PRNG_TYPE_HKDF));

  // Precompute the "a" part of Enc(s << i) and the digits used to generate
  // Enc(s << i).
  ct_pads_.reserve(num_rotations);
  ct_pads_.push_back(std::move(ct_pad));
  ct_sub_pad_digits_.reserve(num_rotations);

  int curr_power = 1;
  int cyclotomic_order = 1 << (log_n + 1);
  for (int i = 1; i < num_rotations; ++i) {
    curr_power = (curr_power * 5) % cyclotomic_order;
    // ct[i-1].a(X^5)
    RLWE_ASSIGN_OR_RETURN(RnsPolynomial prev_sub_a,
                          ct_pads_[i - 1].Substitute(5, rns_moduli_));

    // g^-1(ct[i-1].a(X^5))
    if (prev_sub_a.IsNttForm()) {
      RLWE_RETURN_IF_ERROR(prev_sub_a.ConvertToCoeffForm(rns_moduli_));
    }
    RLWE_ASSIGN_OR_RETURN(auto prev_sub_a_digits,
                          rns_gadget_->Decompose(prev_sub_a, rns_moduli_));
    for (auto& digit : prev_sub_a_digits) {
      RLWE_RETURN_IF_ERROR(digit.ConvertToNttForm(rns_moduli_));
    }

    // g^-1(ct[i-1].a(X^5))^T * gk.a
    RLWE_ASSIGN_OR_RETURN(
        auto curr_a,
        RnsPolynomial::CreateZero(log_n, rns_moduli_, /*is_ntt=*/true));
    for (int j = 0; j < prev_sub_a_digits.size(); ++j) {
      RLWE_RETURN_IF_ERROR(curr_a.FusedMulAddInPlace(prev_sub_a_digits[j],
                                                     gk_pads_[j], rns_moduli_));
    }
    ct_pads_.push_back(std::move(curr_a));
    ct_sub_pad_digits_.push_back(std::move(prev_sub_a_digits));
  }
  return absl::OkStatus();
}

absl::Status WeightMatrix::PreprocessDatabase(
    absl::Span<const RnsPolynomial> pad_rotated_queries0,
    absl::Span<const RnsPolynomial> pad_rotated_queries1) {
  if (pad_rotated_queries0.size() > diagonals00_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "`pad_rotated_queries0` cannot be larger than ", diagonals00_.size()));
  }
  if (pad_rotated_queries1.size() > diagonals01_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "`pad_rotated_queries1` cannot be larger than ", diagonals01_.size()));
  }

  // Precompute ct_pad * diagonals
  RLWE_ASSIGN_OR_RETURN(
      auto pad_inner_product00,
      pad_rotated_queries0[0].Mul(diagonals00_[0], rns_moduli_));
  for (int j = 1; j < pad_rotated_queries0.size(); ++j) {
    RLWE_RETURN_IF_ERROR(pad_inner_product00.FusedMulAddInPlace(
        pad_rotated_queries0[j], diagonals00_[j], rns_moduli_));
  }
  RLWE_ASSIGN_OR_RETURN(
      auto pad_inner_product01,
      pad_rotated_queries1[0].Mul(diagonals01_[0], rns_moduli_));
  for (int j = 1; j < pad_rotated_queries1.size(); ++j) {
    RLWE_RETURN_IF_ERROR(pad_inner_product01.FusedMulAddInPlace(
        pad_rotated_queries1[j], diagonals01_[j], rns_moduli_));
  }
  RLWE_ASSIGN_OR_RETURN(
      auto pad_inner_product10,
      pad_rotated_queries0[0].Mul(diagonals10_[0], rns_moduli_));
  for (int j = 1; j < pad_rotated_queries0.size(); ++j) {
    RLWE_RETURN_IF_ERROR(pad_inner_product10.FusedMulAddInPlace(
        pad_rotated_queries0[j], diagonals10_[j], rns_moduli_));
  }
  RLWE_ASSIGN_OR_RETURN(
      auto pad_inner_product11,
      pad_rotated_queries1[0].Mul(diagonals11_[0], rns_moduli_));
  for (int j = 1; j < pad_rotated_queries1.size(); ++j) {
    RLWE_RETURN_IF_ERROR(pad_inner_product11.FusedMulAddInPlace(
        pad_rotated_queries1[j], diagonals11_[j], rns_moduli_));
  }
  pad_inner_product00_ =
      std::make_unique<RnsPolynomial>(std::move(pad_inner_product00));
  pad_inner_product01_ =
      std::make_unique<RnsPolynomial>(std::move(pad_inner_product01));
  pad_inner_product10_ =
      std::make_unique<RnsPolynomial>(std::move(pad_inner_product10));
  pad_inner_product11_ =
      std::make_unique<RnsPolynomial>(std::move(pad_inner_product11));

  return absl::OkStatus();
}

absl::Status WeightMatrix::PreprocessBlocksForStrassen() {
  // Precompute some linear combinations of blocks of M that are needed by
  // Strassen over 2 x 2 blocks.
  m00_m11_diagonals_.reserve(diagonals00_.size());
  for (int i = 0; i < diagonals00_.size(); ++i) {
    RLWE_ASSIGN_OR_RETURN(auto p,
                          diagonals00_[i].Add(diagonals11_[i], rns_moduli_));
    m00_m11_diagonals_.push_back(std::move(p));
  }

  m10_m11_diagonals_.reserve(diagonals10_.size());
  for (int i = 0; i < diagonals10_.size(); ++i) {
    RLWE_ASSIGN_OR_RETURN(auto p,
                          diagonals10_[i].Add(diagonals11_[i], rns_moduli_));
    m10_m11_diagonals_.push_back(std::move(p));
  }

  m00_m01_diagonals_.reserve(diagonals00_.size());
  for (int i = 0; i < diagonals00_.size(); ++i) {
    RLWE_ASSIGN_OR_RETURN(auto p,
                          diagonals00_[i].Add(diagonals01_[i], rns_moduli_));
    m00_m01_diagonals_.push_back(std::move(p));
  }

  m10_m00_diagonals_.reserve(diagonals10_.size());
  for (int i = 0; i < diagonals10_.size(); ++i) {
    RLWE_ASSIGN_OR_RETURN(auto p,
                          diagonals10_[i].Sub(diagonals00_[i], rns_moduli_));
    m10_m00_diagonals_.push_back(std::move(p));
  }

  m01_m11_diagonals_.reserve(diagonals01_.size());
  for (int i = 0; i < diagonals01_.size(); ++i) {
    RLWE_ASSIGN_OR_RETURN(auto p,
                          diagonals01_[i].Sub(diagonals11_[i], rns_moduli_));
    m01_m11_diagonals_.push_back(std::move(p));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<WeightMatrix::RnsCiphertext>>
WeightMatrix::InnerProductWith(
    absl::Span<const RnsCiphertext> ct_rotated_queries0,
    absl::Span<const RnsCiphertext> ct_rotated_queries1) const {
  if (ct_rotated_queries0.size() > diagonals00_.size()) {
    return absl::InvalidArgumentError(
        "`ct_rotated_queries0` contains too many ciphertexts.");
  }
  if (ct_rotated_queries1.size() > diagonals01_.size()) {
    return absl::InvalidArgumentError(
        "`ct_rotated_queries1` contains too many ciphertexts.");
  }

  std::vector<RnsCiphertext> ct_inner_products;
  RLWE_ASSIGN_OR_RETURN(RnsCiphertext ct_inner_product0,
                        ct_rotated_queries0[0].AbsorbSimple(diagonals00_[0]));
  for (int j = 1; j < ct_rotated_queries0.size(); ++j) {
    RLWE_RETURN_IF_ERROR(ct_inner_product0.FusedAbsorbAddInPlace(
        ct_rotated_queries0[j], diagonals00_[j]));
  }
  RLWE_ASSIGN_OR_RETURN(RnsCiphertext ct_inner_product1,
                        ct_rotated_queries1[0].AbsorbSimple(diagonals01_[0]));
  for (int j = 1; j < ct_rotated_queries1.size(); ++j) {
    RLWE_RETURN_IF_ERROR(ct_inner_product1.FusedAbsorbAddInPlace(
        ct_rotated_queries1[j], diagonals01_[j]));
  }
  RLWE_RETURN_IF_ERROR(ct_inner_product0.AddInPlace(ct_inner_product1));
  ct_inner_products.push_back(std::move(ct_inner_product0));

  RLWE_ASSIGN_OR_RETURN(ct_inner_product0,
                        ct_rotated_queries0[0].AbsorbSimple(diagonals10_[0]));
  for (int j = 1; j < ct_rotated_queries0.size(); ++j) {
    RLWE_RETURN_IF_ERROR(ct_inner_product0.FusedAbsorbAddInPlace(
        ct_rotated_queries0[j], diagonals10_[j]));
  }
  RLWE_ASSIGN_OR_RETURN(ct_inner_product1,
                        ct_rotated_queries1[0].AbsorbSimple(diagonals11_[0]));
  for (int j = 1; j < ct_rotated_queries1.size(); ++j) {
    RLWE_RETURN_IF_ERROR(ct_inner_product1.FusedAbsorbAddInPlace(
        ct_rotated_queries1[j], diagonals11_[j]));
  }
  RLWE_RETURN_IF_ERROR(ct_inner_product0.AddInPlace(ct_inner_product1));
  ct_inner_products.push_back(std::move(ct_inner_product0));

  return ct_inner_products;
}

absl::StatusOr<std::vector<WeightMatrix::RnsCiphertext>>
WeightMatrix::InnerProductWithPreprocessedPads(
    const std::vector<RnsCiphertext>& ct_rotated_queries0,
    const std::vector<RnsCiphertext>& ct_rotated_queries1) const {
  if (ct_rotated_queries0.size() > diagonals00_.size()) {
    return absl::InvalidArgumentError(
        "`ct_rotated_queries0` contains too many ciphertexts.");
  }
  if (ct_rotated_queries1.size() > diagonals01_.size()) {
    return absl::InvalidArgumentError(
        "`ct_rotated_queries1` contains too many ciphertexts.");
  }

  std::vector<RnsCiphertext> ct_inner_products;
  auto error_params = ct_rotated_queries0[0].ErrorParams();
  auto rns_context = ct_rotated_queries0[0].Context();
  // M00 * v0
  RnsCiphertext ct_inner_product0(
      RnsCiphertext::CreateZero(rns_moduli_, error_params, rns_context));
  for (int j = 0; j < ct_rotated_queries0.size(); ++j) {
    RLWE_RETURN_IF_ERROR(
        ct_inner_product0.FusedAbsorbAddInPlaceWithoutPadLazily(
            ct_rotated_queries0[j], diagonals00_[j]));
  }
  RLWE_RETURN_IF_ERROR(ct_inner_product0.MergeLazyOperations());
  RLWE_RETURN_IF_ERROR(
      ct_inner_product0.SetPadComponent(*pad_inner_product00_));

  // M01 * v1
  RnsCiphertext ct_inner_product1(
      RnsCiphertext::CreateZero(rns_moduli_, error_params, rns_context));
  for (int j = 0; j < ct_rotated_queries1.size(); ++j) {
    RLWE_RETURN_IF_ERROR(
        ct_inner_product1.FusedAbsorbAddInPlaceWithoutPadLazily(
            ct_rotated_queries1[j], diagonals01_[j]));
  }
  RLWE_RETURN_IF_ERROR(ct_inner_product1.MergeLazyOperations());
  RLWE_RETURN_IF_ERROR(
      ct_inner_product1.SetPadComponent(*pad_inner_product01_));
  // ip0 = M00 * v0 + M01 * v1
  RLWE_RETURN_IF_ERROR(ct_inner_product0.AddInPlace(ct_inner_product1));
  ct_inner_products.push_back(std::move(ct_inner_product0));

  // M10 * v0
  ct_inner_product0 = RnsCiphertext(
      RnsCiphertext::CreateZero(rns_moduli_, error_params, rns_context));
  for (int j = 0; j < ct_rotated_queries0.size(); ++j) {
    RLWE_RETURN_IF_ERROR(
        ct_inner_product0.FusedAbsorbAddInPlaceWithoutPadLazily(
            ct_rotated_queries0[j], diagonals10_[j]));
  }
  RLWE_RETURN_IF_ERROR(ct_inner_product0.MergeLazyOperations());
  RLWE_RETURN_IF_ERROR(
      ct_inner_product0.SetPadComponent(*pad_inner_product10_));
  // M11 * v1
  ct_inner_product1 = RnsCiphertext(
      RnsCiphertext::CreateZero(rns_moduli_, error_params, rns_context));
  for (int j = 0; j < ct_rotated_queries1.size(); ++j) {
    RLWE_RETURN_IF_ERROR(
        ct_inner_product1.FusedAbsorbAddInPlaceWithoutPadLazily(
            ct_rotated_queries1[j], diagonals11_[j]));
  }
  RLWE_RETURN_IF_ERROR(ct_inner_product1.MergeLazyOperations());
  RLWE_RETURN_IF_ERROR(
      ct_inner_product1.SetPadComponent(*pad_inner_product11_));
  // ip1 = M10 * v0 + M11 * v1
  RLWE_RETURN_IF_ERROR(ct_inner_product0.AddInPlace(ct_inner_product1));
  ct_inner_products.push_back(std::move(ct_inner_product0));

  return ct_inner_products;
}

absl::StatusOr<std::vector<std::vector<WeightMatrix::RnsCiphertext>>>
WeightMatrix::InnerProductWithMany(
    const std::vector<std::vector<std::vector<RnsCiphertext>>>&
        ct_rotated_queries) const {
  int num_queries = ct_rotated_queries.size();
  std::vector<std::vector<RnsCiphertext>> ct_inner_products(num_queries);
  for (int i = 0; i < num_queries; ++i) {
    RLWE_ASSIGN_OR_RETURN(
        ct_inner_products[i],
        InnerProductWithPreprocessedPads(ct_rotated_queries[i][0],
                                         ct_rotated_queries[i][1]));
  }
  return ct_inner_products;
}

absl::StatusOr<std::vector<std::vector<WeightMatrix::RnsCiphertext>>>
WeightMatrix::GenerateRotatedVectors(absl::Span<const RnsCiphertext> ct_vs,
                                     absl::Span<const RnsGaloisKey> gks,
                                     int num_rotations) const {
  if (!IsPreprocessedPad()) {
    return absl::FailedPreconditionError("must call PreprocessPad first");
  }

  constexpr int power = 5;
  int num_keys = gks.size();
  std::vector<std::vector<RnsCiphertext>> ct_rotated_vs(num_rotations);
  ct_rotated_vs[0].reserve(num_keys);
  for (int j = 0; j < num_keys; ++j) {
    ct_rotated_vs[0].push_back(ct_vs[j]);  // first diagonal
  }
  for (int i = 1; i < num_rotations; ++i) {
    ct_rotated_vs[i].reserve(num_keys);
    for (int j = 0; j < num_keys; ++j) {
      RLWE_ASSIGN_OR_RETURN(auto ct_v_sub,
                            ct_rotated_vs[i - 1][j].Substitute(power));
      RLWE_ASSIGN_OR_RETURN(
          auto ct_rotated_v,
          gks[j].ApplyToWithRandomPad(ct_v_sub, ct_sub_pad_digits_[i - 1],
                                      ct_pads_[i]));
      ct_rotated_vs[i].push_back(std::move(ct_rotated_v));
    }
  }
  return ct_rotated_vs;
}

absl::StatusOr<WeightMatrix::StrassenResult> WeightMatrix::Strassen(
    const std::vector<std::vector<RnsCiphertext>>& ct_vs,
    const std::vector<std::vector<RnsGaloisKey>>& gks,
    int num_rotations) const {
  if (ct_vs.size() != 2 || gks.size() != 2) {
    return absl::InvalidArgumentError("`ct_vs` and `gks` must have size 2");
  }
  int k = ct_vs[0].size() / 2;
  if (ct_vs[0].size() != 2 * k || ct_vs[1].size() != 2 * k) {
    return absl::InvalidArgumentError("`ct_vs[i]` size must be 2*k");
  }
  if (gks[0].size() != 2 * k || gks[1].size() != 2 * k) {
    return absl::InvalidArgumentError("`gks[i]` sizes must be 2*k");
  }
  if (!IsPreprocessedDatabase()) {
    return absl::InvalidArgumentError("Database must be preprocessed");
  }
  if (!IsPreprocessedBlocksForStrassen()) {
    return absl::InvalidArgumentError("Database blocks must be preprocessed");
  }
  if (num_rotations < 0 || num_rotations > diagonals00_.size() ||
      num_rotations > diagonals11_.size() ||
      num_rotations > m00_m11_diagonals_.size() ||
      num_rotations > m10_m11_diagonals_.size() ||
      num_rotations > m00_m01_diagonals_.size() ||
      num_rotations > m10_m00_diagonals_.size() ||
      num_rotations > m01_m11_diagonals_.size()) {
    return absl::InvalidArgumentError("`num_rotations` is out of range.");
  }

  // Perform rotations and FMAs one step at a time to reduce memory pressure.
  std::vector<RnsCiphertext> curr_v0 = ct_vs[0];
  std::vector<RnsCiphertext> curr_v1 = ct_vs[1];

  auto error_params = ct_vs[0][0].ErrorParams();
  auto rns_context = ct_vs[0][0].Context();
  RnsCiphertext ct_zero(
      RnsCiphertext::CreateZero(rns_moduli_, error_params, rns_context));
  int num_cols = k;

  std::vector<RnsCiphertext> ct_w1(num_cols, ct_zero), ct_w2(num_cols, ct_zero),
      ct_w3(num_cols, ct_zero), ct_w4(num_cols, ct_zero),
      ct_w5(num_cols, ct_zero), ct_w6(num_cols, ct_zero),
      ct_w7(num_cols, ct_zero);

  constexpr int power = 5;
  for (int i = 0; i < num_rotations; ++i) {
    const auto& d00_i = diagonals00_[i];
    const auto& d11_i = diagonals11_[i];
    const auto& m00_m11_i = m00_m11_diagonals_[i];
    const auto& m10_m11_i = m10_m11_diagonals_[i];
    const auto& m00_m01_i = m00_m01_diagonals_[i];
    const auto& m10_m00_i = m10_m00_diagonals_[i];
    const auto& m01_m11_i = m01_m11_diagonals_[i];

    for (int j = 0; j < num_cols; ++j) {
      const auto& v00 = curr_v0[j];
      const auto& v01 = curr_v0[j + num_cols];
      const auto& v10 = curr_v1[j];
      const auto& v11 = curr_v1[j + num_cols];

      // W1 = (M00 + M11) * (v00 + v11)
      RLWE_RETURN_IF_ERROR(ct_w1[j].FusedAbsorbSumAddInPlaceWithoutPadLazily(
          v00, v11, m00_m11_i));

      // W2 = (M10 + M11) * v00
      RLWE_RETURN_IF_ERROR(
          ct_w2[j].FusedAbsorbAddInPlaceWithoutPadLazily(v00, m10_m11_i));

      // W3 = M00 * (v01 - v11)
      RLWE_RETURN_IF_ERROR(
          ct_w3[j].FusedAbsorbDiffAddInPlaceWithoutPadLazily(v01, v11, d00_i));

      // W4 = M11 * (v10 - v00)
      RLWE_RETURN_IF_ERROR(
          ct_w4[j].FusedAbsorbDiffAddInPlaceWithoutPadLazily(v10, v00, d11_i));

      // W5 = (M00 + M01) * v11
      RLWE_RETURN_IF_ERROR(
          ct_w5[j].FusedAbsorbAddInPlaceWithoutPadLazily(v11, m00_m01_i));

      // W6 = (M10 - M00) * (v00 + v01)
      RLWE_RETURN_IF_ERROR(ct_w6[j].FusedAbsorbSumAddInPlaceWithoutPadLazily(
          v00, v01, m10_m00_i));

      // W7 = (M01 - M11) * (v10 + v11)
      RLWE_RETURN_IF_ERROR(ct_w7[j].FusedAbsorbSumAddInPlaceWithoutPadLazily(
          v10, v11, m01_m11_i));
    }

    if (i < num_rotations - 1) {
      for (int j = 0; j < 2 * num_cols; ++j) {
        RLWE_ASSIGN_OR_RETURN(auto v0_sub, curr_v0[j].Substitute(power));
        RLWE_ASSIGN_OR_RETURN(
            curr_v0[j], gks[0][j].ApplyToWithRandomPad(
                            v0_sub, ct_sub_pad_digits_[i], ct_pads_[i + 1]));
        RLWE_ASSIGN_OR_RETURN(auto v1_sub, curr_v1[j].Substitute(power));
        RLWE_ASSIGN_OR_RETURN(
            curr_v1[j], gks[1][j].ApplyToWithRandomPad(
                            v1_sub, ct_sub_pad_digits_[i], ct_pads_[i + 1]));
      }
    }
  }

  // Merge lazy operations for all W blocks.
  for (int j = 0; j < num_cols; ++j) {
    RLWE_RETURN_IF_ERROR(ct_w1[j].MergeLazyOperations());
    RLWE_RETURN_IF_ERROR(ct_w2[j].MergeLazyOperations());
    RLWE_RETURN_IF_ERROR(ct_w3[j].MergeLazyOperations());
    RLWE_RETURN_IF_ERROR(ct_w4[j].MergeLazyOperations());
    RLWE_RETURN_IF_ERROR(ct_w5[j].MergeLazyOperations());
    RLWE_RETURN_IF_ERROR(ct_w6[j].MergeLazyOperations());
    RLWE_RETURN_IF_ERROR(ct_w7[j].MergeLazyOperations());
  }

  StrassenResult result;
  result.u00.reserve(num_cols);
  result.u01.reserve(num_cols);
  result.u10.reserve(num_cols);
  result.u11.reserve(num_cols);

  for (int j = 0; j < num_cols; ++j) {
    // U00 = W1 + W4 - W5 + W7
    // The W terms are already merged, so additions/subtractions are simpler.
    RnsCiphertext u00 = ct_w1[j];
    RLWE_RETURN_IF_ERROR(u00.AddInPlaceWithoutPad(ct_w4[j]));
    RLWE_RETURN_IF_ERROR(u00.SubInPlaceWithoutPad(ct_w5[j]));
    RLWE_RETURN_IF_ERROR(u00.AddInPlaceWithoutPad(ct_w7[j]));
    // No need for MergeLazyOperations() here, as all W terms are already
    // merged.
    RLWE_ASSIGN_OR_RETURN(RnsPolynomial p00, u00.Component(0));
    result.u00.push_back(std::move(p00));

    // U01 = W3 + W5
    RnsCiphertext u01 = ct_w3[j];
    RLWE_RETURN_IF_ERROR(u01.AddInPlaceWithoutPad(ct_w5[j]));
    // No MergeLazyOperations() needed here.
    RLWE_ASSIGN_OR_RETURN(RnsPolynomial p01, u01.Component(0));
    result.u01.push_back(std::move(p01));

    // U10 = W2 + W4
    RnsCiphertext u10 = ct_w2[j];
    RLWE_RETURN_IF_ERROR(u10.AddInPlaceWithoutPad(ct_w4[j]));
    // No MergeLazyOperations() needed here.
    RLWE_ASSIGN_OR_RETURN(RnsPolynomial p10, u10.Component(0));
    result.u10.push_back(std::move(p10));

    // U11 = W1 - W2 + W3 + W6
    RnsCiphertext u11 = ct_w1[j];
    RLWE_RETURN_IF_ERROR(u11.SubInPlaceWithoutPad(ct_w2[j]));
    RLWE_RETURN_IF_ERROR(u11.AddInPlaceWithoutPad(ct_w3[j]));
    RLWE_RETURN_IF_ERROR(u11.AddInPlaceWithoutPad(ct_w6[j]));
    // No MergeLazyOperations() needed here.
    RLWE_ASSIGN_OR_RETURN(RnsPolynomial p11, u11.Component(0));
    result.u11.push_back(std::move(p11));
  }

  result.pad00 = pad_inner_product00_.get();
  result.pad01 = pad_inner_product01_.get();
  result.pad10 = pad_inner_product10_.get();
  result.pad11 = pad_inner_product11_.get();
  return result;
}

absl::StatusOr<WeightMatrix::RnsPolynomial> WeightMatrix::RawDecrypt(
    const RnsPolynomial& key, const RnsCiphertext& ct) const {
  RnsPolynomial s_power = key;
  RLWE_ASSIGN_OR_RETURN(RnsPolynomial output,
                        RnsPolynomial::CreateZero(key.LogN(), rns_moduli_));
  int ct_len = ct.Len();
  for (int i = 0; i < ct_len; ++i) {
    // Get the i-th component
    RLWE_ASSIGN_OR_RETURN(RnsPolynomial ci, ct.Component(i));

    // Compute the next power of the secret polynomial s.
    if (i > 1) {
      RLWE_RETURN_IF_ERROR(s_power.MulInPlace(key, rns_moduli_));
    }
    // Compute c[i] * s^i.
    if (i > 0) {
      RLWE_RETURN_IF_ERROR(ci.MulInPlace(s_power, rns_moduli_));
    }
    // Add c[i] * s^i to the result.
    RLWE_RETURN_IF_ERROR(output.AddInPlace(ci, rns_moduli_));
  }
  return output;
}

}  // namespace private_inference

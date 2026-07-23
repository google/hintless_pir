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

#ifndef HINTLESS_PIR_PRIVATE_INFERENCE_WEIGHT_MATRIX_H_
#define HINTLESS_PIR_PRIVATE_INFERENCE_WEIGHT_MATRIX_H_
// This file implements a weight matrix for a model's linear layer, and the
// homomorphic matrix-vector product with an encrypted vector. More
// specifically, the query vector is encrypted under RLWE for which the client
// holds the secret key, and the random "a" component of the encrypted vector is
// public and shared among all clients. We also implement a batched homomorphic
// matrix-vector multiplication with multiple encrypted query vectors from
// potentially different clients, using the Strassen method to speedup the
// computation.
//
// The WeightMatrix class can be used to implement the linear layer in a
// private inference protocol, where the inference server holds the weight
// matrix and the client wants to obtain the product with its query vector
// without revealing the query vector to the server.

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "private_inference/parameters.h"
#include "shell_encryption/montgomery.h"
#include "shell_encryption/rns/finite_field_encoder.h"
#include "shell_encryption/rns/rns_bfv_ciphertext.h"
#include "shell_encryption/rns/rns_context.h"
#include "shell_encryption/rns/rns_galois_key.h"
#include "shell_encryption/rns/rns_modulus.h"
#include "shell_encryption/rns/rns_polynomial.h"

namespace private_inference {

class WeightMatrix {
 public:
  using ModularInt = rlwe::MontgomeryInt<Integer>;
  using RnsContext = rlwe::RnsContext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using RnsCiphertext = rlwe::RnsBfvCiphertext<ModularInt>;
  using RnsGadget = rlwe::RnsGadget<ModularInt>;
  using RnsGaloisKey = rlwe::RnsGaloisKey<ModularInt>;
  using PrimeModulus = rlwe::PrimeModulus<ModularInt>;
  using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;
  using EncryptedVector = std::vector<const RnsCiphertext>;

  // Create an empty matrix, where `rns_context` and `rns_gadget` provide the
  // underlying RLWE parameters and gadget, and `prng_seed_ct_pad` and
  // `prng_seed_gk_pad` are the seeds for the "a" element of the encrypted
  // vector and the Galois keys to be used in all queries.
  static absl::StatusOr<std::unique_ptr<WeightMatrix>> Create(
      const RnsContext* rns_context, const RnsGadget* rns_gadget = nullptr,
      const absl::string_view prng_seed_ct_pad = "",
      const absl::string_view prng_seed_gk_pad = "");

  // Set matrix data
  absl::Status SetData(const std::vector<std::vector<Integer>>& data);

  // Preprocess the rotated ciphertext "a" elements.
  absl::Status PreprocessPad(int num_rotations);

  // Preprocess the database with the given random pads to speedup inner product
  // computation when query is available.
  absl::Status PreprocessDatabase(
      absl::Span<const RnsPolynomial> pad_rotated_queries0,
      absl::Span<const RnsPolynomial> pad_rotated_queries1);

  absl::Status PreprocessBlocksForStrassen();

  // Compute the matrix-vector product with an encrypted vector. We assume the
  // vector v is split into two halves v0 and v1, where `ct_rotated_queries0` is
  // a collection of rotated ciphertexts of the form Enc(v0 << i) for i = 0 ..
  // num_rotations, and similarly for `ct_rotated_queries1`.
  // Returns a collection of ciphertexts representing the blocks of the
  // matrix-vector product.
  absl::StatusOr<std::vector<RnsCiphertext>> InnerProductWith(
      absl::Span<const RnsCiphertext> ct_rotated_queries0,
      absl::Span<const RnsCiphertext> ct_rotated_queries1) const;

  // Compute the matrix-vector product with an encrypted vector, where the
  // matrix has been preprocessed. As in `InnerProductWith`, we assume the
  // vector v is split into two halves v0 and v1, where `ct_rotated_queries0` is
  // a collection of rotated ciphertexts of the form Enc(v0 << i) for i = 0 ..
  // num_rotations, and similarly for `ct_rotated_queries1`.
  // Returns a collection of ciphertexts representing the blocks of the
  // matrix-vector product.
  // Returns error if `PreprocessDatabase()` has not been called.
  absl::StatusOr<std::vector<RnsCiphertext>> InnerProductWithPreprocessedPads(
      const std::vector<RnsCiphertext>& ct_rotated_queries0,
      const std::vector<RnsCiphertext>& ct_rotated_queries1) const;

  // Compute the matrix-matrix product M * V, where each of `ct_rotated_queries`
  // is a collection of rotated columns of V.
  // Returns a vector of ciphertexts, where each encrypts a column of M * V.
  absl::StatusOr<std::vector<std::vector<RnsCiphertext>>> InnerProductWithMany(
      const std::vector<std::vector<std::vector<RnsCiphertext>>>&
          ct_rotated_queries) const;

  // Return Enc(v << j) for j = 0..num_rotations for all Enc(v) in ct_vs.
  absl::StatusOr<std::vector<std::vector<RnsCiphertext>>>
  GenerateRotatedVectors(absl::Span<const RnsCiphertext> ct_vs,
                         absl::Span<const RnsGaloisKey> gks,
                         int num_rotations) const;

  // Captures the return values of Strassen().
  struct StrassenResult {
    std::vector<RnsPolynomial> u00;
    std::vector<RnsPolynomial> u01;
    std::vector<RnsPolynomial> u10;
    std::vector<RnsPolynomial> u11;

    RnsPolynomial* pad00;
    RnsPolynomial* pad01;
    RnsPolynomial* pad10;
    RnsPolynomial* pad11;
  };

  // One level Strassen.
  // We compute product M * Enc(V) = ( M00 M01 ) * ( C00 C01 )
  //                                 ( M10 M11 )   ( C10 C11 )
  // where Mij is stored in diagonalsij_, and Cij is 1 x K that contains K
  // ciphertexts a * s_ijk + e_ijk + Encode(V_ijk).
  // We assume the ciphertexts in the same column are from the same client, i.e.
  // C00[k] and C10[k] are both from client k, and C10[k] and C11[k] are both
  // from client K + k.
  // The results are ciphertexts of the form
  // - for the first k columns: a' * s_00k + a'' * s10k + error + product;
  // - for the other k columns: a' * s_10k + a'' * s11k + error + product.
  absl::StatusOr<StrassenResult> Strassen(
      const std::vector<std::vector<RnsCiphertext>>& ct_vs,
      const std::vector<std::vector<RnsGaloisKey>>& gks,
      int num_rotations) const;

  // Accessors
  bool IsPreprocessedPad() const {
    return !ct_pads_.empty() && !ct_sub_pad_digits_.empty() &&
           !gk_pads_.empty();
  }

  bool IsPreprocessedDatabase() const {
    return pad_inner_product00_ != nullptr && pad_inner_product01_ != nullptr &&
           pad_inner_product10_ != nullptr && pad_inner_product11_ != nullptr;
  }

  bool IsPreprocessedBlocksForStrassen() const {
    return !m00_m11_diagonals_.empty() && !m10_m11_diagonals_.empty() &&
           !m00_m01_diagonals_.empty() && !m10_m00_diagonals_.empty() &&
           !m01_m11_diagonals_.empty();
  }

  // The PRNG seed used to generate the "a" element of Enc(query).
  absl::string_view PrngSeedCtPad() const { return prng_seed_ct_pad_; }

  // The PRNG seed used to generate the "a" elements of a Galois key.
  absl::string_view PrngSeedGkPad() const { return prng_seed_gk_pad_; }

  // The "a" element of ciphertexts Enc(query << i) for i = 0 .. num_rotations.
  // Note that all query ciphertexts Enc(query) have the same "a" element
  // generated from PrngSeedCtPad().
  const std::vector<RnsPolynomial>& PreprocessedRotatedCiphertextPads() const {
    return ct_pads_;
  }

  const std::vector<std::vector<RnsPolynomial>>&
  PreprocessedSubCiphertextPadDigits() const {
    return ct_sub_pad_digits_;
  }

  absl::StatusOr<RnsPolynomial> RawDecrypt(const RnsPolynomial& key,
                                           const RnsCiphertext& ct) const;

 private:
  explicit WeightMatrix(const RnsContext* rns_context,
                        std::vector<const PrimeModulus*> rns_moduli,
                        const RnsGadget* rns_gadget, Encoder encoder,
                        std::string prng_seed_ct_pad,
                        std::string prng_seed_gk_pad)
      : rns_context_(rns_context),
        rns_moduli_(std::move(rns_moduli)),
        rns_gadget_(rns_gadget),
        encoder_(std::move(encoder)),
        prng_seed_ct_pad_(std::move(prng_seed_ct_pad)),
        prng_seed_gk_pad_(std::move(prng_seed_gk_pad)) {}

  const RnsContext* rns_context_;
  const std::vector<const PrimeModulus*> rns_moduli_;
  const RnsGadget* rns_gadget_;
  const Encoder encoder_;

  // A matrix is divided into 2x2 blocks, each block is stored using diagonals.
  std::vector<RnsPolynomial> diagonals00_;
  std::vector<RnsPolynomial> diagonals01_;
  std::vector<RnsPolynomial> diagonals10_;
  std::vector<RnsPolynomial> diagonals11_;

  std::vector<RnsPolynomial> m00_m11_diagonals_;
  std::vector<RnsPolynomial> m10_m11_diagonals_;
  std::vector<RnsPolynomial> m00_m01_diagonals_;
  std::vector<RnsPolynomial> m10_m00_diagonals_;
  std::vector<RnsPolynomial> m01_m11_diagonals_;

  // The random pads, i.e. the "a" parts, of the ciphertexts encrypting the
  // matrix-vector products between the blocks of diagonals and the query vector
  std::unique_ptr<RnsPolynomial> pad_inner_product00_;
  std::unique_ptr<RnsPolynomial> pad_inner_product01_;
  std::unique_ptr<RnsPolynomial> pad_inner_product10_;
  std::unique_ptr<RnsPolynomial> pad_inner_product11_;

  // FIXME: The followings are preprocessed "a" or pad parts of rotated
  // ciphertexts. They should really be in the server class.
  std::string prng_seed_ct_pad_;
  std::string prng_seed_gk_pad_;

  std::vector<RnsPolynomial> ct_pads_;
  std::vector<std::vector<RnsPolynomial>> ct_sub_pad_digits_;
  std::vector<RnsPolynomial> gk_pads_;
};

template <typename E>
class BlockReader {
 public:
  explicit BlockReader(int n_rows, int n_cols, int row_start, int col_start,
                       int d)
      : n_rows_(n_rows),
        n_cols_(n_cols),
        row_start_(row_start),
        col_start_(col_start),
        d_(d) {}

  E GetDiagonal(int i, int j, const std::vector<std::vector<E>>& data) const {
    int row = j;
    int col = (i + j) % d_;
    if (row >= n_rows_ || col >= n_cols_) {
      return E();
    }
    return data[row_start_ + row][col_start_ + col];
  }

 private:
  int n_rows_;     // # rows in the block
  int n_cols_;     // # columns in the block
  int row_start_;  // row starting index
  int col_start_;  // column starting index
  int d_;          // length of diagonal
};

}  // namespace private_inference

#endif  // HINTLESS_PIR_PRIVATE_INFERENCE_WEIGHT_MATRIX_H_

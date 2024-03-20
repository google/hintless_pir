# Hintless Single-Server Private Information Retrieval

This library contains implementations of the single-server private information
retrieval (PIR) protocols of the following paper:

> Li B., Micciancio D., Raykova M., Schultz-Wu M. (2023). Hintless Single-Server
> Private Information Retrieval, IACR ePrint https://eprint.iacr.org/2023/1733

## About the hintless PIR protocols

A Private Information Retrieval (PIR) protocol enables a client to retrieve a
record from a large public database hosted on server(s) while hiding the
identity of the retrieved record. In the single-server setting, the client uses
Homomorphic Encryption (HE) to encrypt the index of the desired record, and the
server homomorphically computes on the encrypted query over the database,
returning an encrypted record to the client. When assuming the server is
semi-honest, i.e. the server follows the PIR protocol but may try to infer
passively which record is being selected, semantics security of the HE scheme
ensures that the server learns nothing about the client's desired record.

In the hintless setting, the client does not store any database-dependent state,
and the server does not store any client-dependent information. In other words,
there is no interactive setup phase in hintless PIR protocols. It becomes much
easier to handle database update as the client does not need to be involved to
process database updates.

This library currently contains an implementation of the HintlessPIR protocol,
which is based on the recent LWE-based
[SimplePIR](https://eprint.iacr.org/2022/949) and eliminates its
database-dependent client preprocessing step by outsoucing the "hint" related
computation to the server. The outsoucing computation is done using a RLWE-based
Linear PIR (LinPIR) protocol, which can itself be used as a standalone protocol
to efficiently compute matrix-vector products.

## Building the libraries

This repository requires Bazel. You can install Bazel by following the
instructions for your platform on the
[Bazel website](https://docs.bazel.build/versions/master/install.html).

Once you have installed Bazel you can clone this repository and run all tests
that are included by navigating into the root folder and running:

```bash
bazel test //...
```

More specifically, this library depends on the following projects:

- [`SHELL homomorphic encryption library`](https://github.com/google/shell-encryption)
- [`Eigen linear algebra library`](https://eigen.tuxfamily.org/)
- [`Abseil C++ library`](https://abseil.io/)

## LinPIR

In the LinPIR protocol, the server holds a public matrix $M$ of dimension `m_0`
by `m_1` over $\mathbb{Z}_t$, and the client wants to privately computes the
product $M \cdot \vec{v}$ for a private vector $\vec{v}$ held by the client. To
instantiate the LinPIR protocol, you need to specify its parameters, e.g.:

```c++
auto linpir_params = hintless_pir::linpir::RlweParameters<uint64_t>{
  .log_n = 12,
  .qs = {35184371884033ULL, 35184371703809ULL},
  .ts = {65537},
  .gadget_log_bs = {16, 16},
  .error_variance = 8,
  .prng_type = rlwe::PRNG_TYPE_HKDF,
  .rows_per_block = 512,
};
```

Most of these parameters specify a RLWE-based linear homomorphic encryption
scheme in a ring $R_Q = \mathbb{Z}_Q[X] / (X^N + 1)$:

-   `log_n`: $\log_2(N)$, where $N$ must be a power of 2 and larger than `m_1`;
-   `qs`: the ciphertext moduli, where each member should be a prime factor $q$
    of $Q$ such that $q - 1 \equiv 0 \pmod{2N}$;
-   `ts`: the plaintext moduli, where each member should be a prime $t$ such
    that $t - 1 \equiv 0 \pmod{2N}$. This should be a singleton when
    instantiating LinPIR as a standalone protocol;
-   `gadget_log_bs`: each member specifies the bit size of the gadget base for a
    corresponding prime modulus in `qs`;
-   `error_vatiance`: the variance of a centered binomial distribution, used as
    the security and error distributions for the RLWE-based HE scheme;
-   `prng_type`: the type of PRNG used to sample random elements in the
    RLWE-based HE scheme.

In addition, the parameter `rows_per_block` specifies how many rows of $M$ to be
grouped in a block in the homomorphic computation. The response from the server
contains RLWE ciphertexts where each ciphertext encrypts the product between a
block of $M$ and the client's private vector $\vec{v}$. For more details, see
the above paper, as well as the
[`linpir::Parameters` class documentation](linpir/parameters.h).

## HintlessPIR

To instantiate the HintlessPIR (or hintless SimplePIR), you need to specify the
database configuration and LWE / LinPIR parameters as follows:

```c++
auto params = hintless_pir::hintless_simplepir::Parameters{
    .db_rows = 32768,
    .db_cols = 32768,
    .db_record_bit_size = 8,
    .lwe_secret_dim = 1400,
    .lwe_modulus_bit_size = 32,
    .lwe_plaintext_bit_size = 8,
    .lwe_error_variance = 8,
    .linpir_params =
        hintless_pir::linpir::RlweParameters<uint64_t>{
            .log_n = 12,
            .qs = {35184371884033ULL, 35184371703809ULL},
            .ts = {2056193, 1990657},
            .gadget_log_bs = {16, 16},
            .error_variance = 8,
            .prng_type = rlwe::PRNG_TYPE_HKDF,
            .rows_per_block = 1024,
        },
    .prng_type = rlwe::PRNG_TYPE_HKDF,
};
```

-   `db_rows` and `db_cols`: as in SimplePIR, the product of `db_rows` and
    `db_cols` is the number of records in the database (after possible padding);
-   `db_record_bit_size`: the bit size of each database record;
-   `lwe_secret_dim`: the dimension of client's LWE secret vector, as in
    SimplePIR;
-   `lwe_modulus_bit_size`: the bit size of LWE ciphertext modulus, can be
    either 32 or 64;
-   `lwe_plaintext_bit_size`: the bit size of the LWE plaintext modulus;
-   `lwe_error_variance`: the variance of a centered binomial distribution, used
    as the error distribution of LWE encryption;
-   `linpir_params`: the parameters of the RLWE-based LinPIR protocols used to
    outsource the hint related computation, where we use a LinPIR instance per
    member in `ts`, such that the product of elements in `ts` is an upper bound
    on l_inf norm of the hint computation;
-   `prng_type`: the type of PRNG used to sample random elements in LWE
    encryption.

For more details of the protocol construction and parameter selection, see the
above paper, as well as the
[`hintless_simplepir::Parameters` class documentation](hintless_simplepir/parameters.h).

## Security

To report a security issue, please read [SECURITY.md](SECURITY.md).

## Disclaimer

This is not an officially supported Google product. The code is provided as-is,
with no guarantees of correctness or security.

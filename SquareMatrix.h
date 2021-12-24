//
// Created by argem on 25.11.2021.
//

#ifndef EVM_LAB7_SQUAREMATRIX_H
#define EVM_LAB7_SQUAREMATRIX_H

#include <vector>
#include <memory>
#include <immintrin.h>
#include <iostream>
#include "AlignedAllocator.h"


#define IS_VECTORIZED


template<std::size_t Size, typename Alloc = AlignedAllocator<float, 64ull>>
class SquareMatrix {

private:
    std::vector<float, Alloc> mem_;

public:

    float* data();
    [[nodiscard]] const float* data() const;

    static constexpr std::size_t size_ = Size;
    static constexpr std::size_t aligment_ = Alloc::aligment_;


    SquareMatrix(const SquareMatrix&) = default;
    SquareMatrix& operator = (const SquareMatrix&) = default;
    ~SquareMatrix() = default;

    SquareMatrix();
    SquareMatrix(SquareMatrix&&) noexcept;
    SquareMatrix& operator = (SquareMatrix&&) noexcept;


    float* operator[](size_t) noexcept;
    const float* operator[](size_t) const noexcept;

    static SquareMatrix idMatrix();

    SquareMatrix inverseMatrix(size_t) const;
};



#ifdef IS_VECTORIZED

template<typename T>
std::remove_const_t<std::remove_reference_t<T>> operator * (T&& m, float s){

    using U = std::remove_const_t<std::remove_reference_t<T>>;
    U res = std::forward<T>(m);

    static_assert(U::aligment_%32ull == 0);

    auto mem = res.data();
    auto size = U::size_;

    auto mem_256 = reinterpret_cast<__m256*>(mem);
    auto s_256 = _mm256_set1_ps(s);

    size_t cnt_elements = size*size;
    size_t cnt_iters = cnt_elements/16;
    for(size_t i=0; i<cnt_iters; ++i){

        auto p0 = _mm256_mul_ps(mem_256[i*2], s_256);
        _mm256_store_ps(&mem[i * 16], p0);

        auto p1 = _mm256_mul_ps(mem_256[i*2+1], s_256);
        _mm256_store_ps(&mem[i * 16 + 8], p1);
    }

    for(size_t i=cnt_iters*16; i<cnt_elements; ++i){
        mem[i] *= s;
    }

    return res;
}


template<typename T>
void bullshit_matrix_sum(T& a, const T& b){

    auto size = T::size_;
    auto mem_a = a.data();
    auto mem_b = b.data();

    static_assert(T::aligment_%32ull == 0);
    auto mem256_a = reinterpret_cast<__m256*>(mem_a);
    auto mem256_b = reinterpret_cast<const __m256*>(mem_b);

    size_t cnt_elements = size*size;
    size_t cnt_iters = cnt_elements/16;
    for(size_t i=0; i<cnt_iters; ++i){

        auto p0 = _mm256_add_ps(mem256_a[i*2], mem256_b[i*2]);
        _mm256_store_ps(&mem_a[i * 16], p0);

        auto p1 = _mm256_add_ps(mem256_a[i*2+1], mem256_b[i*2+1]);
        _mm256_store_ps(&mem_a[i * 16 + 8], p1);
    }

    for(size_t i=cnt_iters*16; i<cnt_elements; ++i){
        mem_a[i] += mem_b[i];
    }
}


template<typename T1, typename T2>
std::remove_const_t<std::remove_reference_t<T1>> operator + (T1&& a, T2&& b){

    using T1_ = std::remove_const_t<std::remove_reference_t<T1>>;
    using T2_ = std::remove_const_t<std::remove_reference_t<T2>>;
    static_assert(std::is_same_v<T1_, T2_>);

    if(std::is_same_v<T1_, T1>){

        T1_ res = std::forward<T1>(a);
        bullshit_matrix_sum(res, b);
        return res;
    }
    else{

        T1_ res = std::forward<T2>(b);
        bullshit_matrix_sum(res, a);
        return res;
    }
}


template<std::size_t Size, typename Alloc>
SquareMatrix<Size, Alloc> operator * (const SquareMatrix<Size, Alloc>& a_, const SquareMatrix<Size, Alloc>& b_){

    static_assert(Alloc::aligment_%32ull == 0);

    SquareMatrix<Size, Alloc> res;
    auto mem_a = a_.data();
    auto mem_b = b_.data();
    auto mem_r = res.data();


    size_t cnt_iters = Size/8ull;
    for (size_t i = 0; i < Size; ++i){

        auto r = mem_r + i * Size;

        for (size_t k = 0; k < Size; ++k){

            auto b = mem_b + k * Size;
            auto a = mem_a[i * Size + k];

            for (size_t j = 0; j < cnt_iters; ++j) {

                __m256 temp = _mm256_mul_ps(_mm256_set1_ps(a), _mm256_loadu_ps(b + j*8));
                _mm256_storeu_ps(r + j*8, _mm256_add_ps(_mm256_loadu_ps(r + j*8), temp));
            }

            for(size_t j=cnt_iters*8; j<Size; ++j){
                r[j] += a * b[j];
            }
        }
    }

    return res;
}

#else

template <size_t Size, typename Alloc>
auto operator * (const SquareMatrix<Size, Alloc>& m, float s){

    SquareMatrix<Size, Alloc> res;
    auto mem_res = res.data();
    auto mem_m = m.data();
    for(size_t i=0; i<Size*Size; ++i){
        mem_res[i] = mem_m[i] * s;
    }

    return res;
}


template <size_t Size, typename Alloc>
auto operator * (const SquareMatrix<Size, Alloc>& a, const SquareMatrix<Size, Alloc>& b){

    SquareMatrix<Size, Alloc> res;
    auto mem_res = res.data();
    auto mem_a = a.data();
    auto mem_b = b.data();

    for (size_t i = 0; i < Size; ++i){

        float * r = mem_res + i * Size;
        for (int j = 0; j < Size; ++j) {
            r[j] = 0;
        }

        for (int k = 0; k < Size; ++k){

            const float * b_ = mem_b + k * Size;
            float a_ = mem_a[i*Size + k];
            for (int j = 0; j < Size; ++j) {
                r[j] += a_ * b_[j];
            }
        }
    }

    return res;
}


template <size_t Size, typename Alloc>
auto operator + (const SquareMatrix<Size, Alloc>& a, const SquareMatrix<Size, Alloc>& b){

    SquareMatrix<Size, Alloc> res;
    auto mem_r = res.data();
    auto mem_a = a.data();
    auto mem_b = b.data();

    for(size_t i=0; i<Size*Size; ++i){
        mem_r[i] = mem_a[i] + mem_b[i];
    }

    return res;
}


#endif



template<std::size_t Size, typename Alloc>
SquareMatrix<Size, Alloc>::SquareMatrix(): mem_(Size*Size) {}


template<std::size_t Size, typename Alloc>
SquareMatrix<Size, Alloc>::SquareMatrix(SquareMatrix &&m) noexcept: mem_(std::move(m.mem_)) {}


template<std::size_t Size, typename Alloc>
SquareMatrix<Size, Alloc>& SquareMatrix<Size, Alloc>::operator=(SquareMatrix &&m)  noexcept {

    mem_ = std::move(m.mem_);
    return *this;
}


template<std::size_t Size, typename Alloc>
SquareMatrix<Size, Alloc> SquareMatrix<Size, Alloc>::idMatrix() {

    SquareMatrix id;

    for(size_t i=0; i<Size; ++i)
        id[i][i] = 1.0;

    return id;
}


template<std::size_t Size, typename Alloc>
float *SquareMatrix<Size, Alloc>::operator[](size_t line) noexcept{

    return mem_.data() + Size * line;
}


template<std::size_t Size, typename Alloc>
const float *SquareMatrix<Size, Alloc>::operator[](size_t line) const noexcept {

    return mem_.data() + Size * line;
}


template<std::size_t Size, typename Alloc>
SquareMatrix<Size, Alloc> SquareMatrix<Size, Alloc>::inverseMatrix(size_t cnt_iters) const{

    float mx_col = 0;
    float mx_row = 0;

    for(size_t i = 0; i < Size; ++i){

        float col = 0, row = 0;
        for(size_t j = 0; j < Size; ++j){
            row += std::abs((*this)[i][j]);
            col += std::abs((*this)[j][i]);
        }

        mx_col = std::max(mx_col, col);
        mx_row = std::max(mx_row, row);
    }

    auto res = std::move((*this) * (1/mx_col/mx_row));

    while(cnt_iters--){

        res = res * (idMatrix()*2 + ((*this)*res) * (-1));
    }

    return res;
}


template<std::size_t Size, typename Alloc>
float *SquareMatrix<Size, Alloc>::data() {

    return mem_.data();
}


template<std::size_t Size, typename Alloc>
const float *SquareMatrix<Size, Alloc>::data() const {

    return mem_.data();
}


template<size_t Size, typename Alloc>
std::ostream& operator << (std::ostream& out, const SquareMatrix<Size, Alloc>& m){

    using std::cout, std::endl;

    for(size_t i=0; i<Size; ++i){
        for(size_t j=0; j<Size; ++j){
            cout << m[i][j] << ' ';
        }
        cout << endl;
    }
    cout << endl;

    return out;
}



#endif //EVM_LAB7_SQUAREMATRIX_H

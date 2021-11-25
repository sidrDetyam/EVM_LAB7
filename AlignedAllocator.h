//
// Created by argem on 21.11.2021.
//

#ifndef EVM_LAB7_ALIGNEDALLOCATOR_H
#define EVM_LAB7_ALIGNEDALLOCATOR_H

#include <memory>


template<typename T, std::size_t Aligment>
class AlignedAllocator {

public:

    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef std::true_type propagate_on_container_move_assignment;

    static constexpr std::size_t aligment_ = Aligment;

    template< class U > struct rebind {
        typedef AlignedAllocator<U, Aligment> other;
    };


    AlignedAllocator() = default;
    ~AlignedAllocator() = default;


    [[nodiscard]] std::size_t max_size() const{

        return (static_cast<size_type>(0)-static_cast<size_type>(1)) / sizeof(T);
    }


    T* allocate(std::size_t cnt) const{

        if(cnt == 0ull){
            return nullptr;
        }

        if(cnt > max_size()){
            throw std::bad_alloc();
        }

        void* ptr = _mm_malloc(cnt * sizeof(T), Aligment);

        if(ptr == nullptr){
            throw std::bad_alloc();
        }

        return static_cast<T*>(ptr);
    }


    void deallocate(T* ptr, std::size_t) const{

        return _mm_free(ptr);
    }


    bool operator ==(const AlignedAllocator&) const{
        return true;
    }


    bool operator !=(const AlignedAllocator&) const{
        return false;
    }
};


#endif //EVM_LAB7_ALIGNEDALLOCATOR_H

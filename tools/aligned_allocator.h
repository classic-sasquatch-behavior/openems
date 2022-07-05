/*
*	Copyright (C) 2010 Sebastian Held (sebastian.held@gmx.de)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// based on http://blogs.msdn.com/b/vcblog/archive/2008/08/28/the-aligned_allocator.aspx
// from Stephan T. Lavavej


// The following headers are required for all allocators.
#include <cstdlib>
#include <cstddef>
#include <limits>
#include <new>       // Required for placement new and std::bad_alloc
#include <stdexcept> // Required for std::length_error

template <typename T, std::align_val_t N> struct aligned_allocator
{
	// The following will be the same for virtually all allocators.
	using pointer = T*;
	using const_pointer = const T*;
	using const_reference = const T&;
	using value_type = T;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;

    template <typename U> struct rebind
    {
		using other = aligned_allocator<U, N>;
	};

	// Default constructor, copy constructor, rebinding constructor, and destructor.
	// Empty for stateless allocators.
	aligned_allocator() = default;
	template <typename U> constexpr aligned_allocator(const aligned_allocator<U, N>&) noexcept {}

	// The following will be different for each allocator.
	[[nodiscard]] T* allocate(const std::size_t n) const
	{
		// The return value of allocate(0) is unspecified.
		// aligned_allocator returns NULL in order to avoid depending
		// on malloc(0)'s implementation-defined behavior
		// (the implementation can define malloc(0) to return NULL,
		// in which case the bad_alloc check below would fire).
		// All allocators can return NULL in this case.
		if (n == 0)
		{
			return NULL;
		}

		// All allocators should contain an integer overflow check.
		// The Standardization Committee recommends that std::length_error
		// be thrown in the case of integer overflow.
		if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
			throw std::bad_array_new_length {};
		}

		// Allocators should throw std::bad_alloc in the case of memory allocation failure.
		if (auto p = static_cast<T*>(std::aligned_alloc(static_cast<std::size_t>(N), n*sizeof(T)))) {
			return p;
		}

		throw std::bad_alloc {};
	}

	void deallocate(T* const p, [[maybe_unused]] const size_t n) const
	{
		std::free(p);
	}
};

template <class T, class U, std::align_val_t NT, std::align_val_t NU>
constexpr bool operator==(const aligned_allocator<T, NT>&, const aligned_allocator<U, NU>&) { return true; }

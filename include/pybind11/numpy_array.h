/*
	pybind11/numpy_array.h: wrapper for numpy array providing a STL compliant interface

	Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

	All rights reserved. Use of this source code is governed by a
	BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include "numpy.h"
#include <memory>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif


NAMESPACE_BEGIN(pybind11)

// Note: this is a temporary version of the wrapper, supporting
// one-dimensional array only. This will be improved to support
// N dimensional arrays in the next iteration

template <class Derived,
		  class T,
		  class Distance = std::ptrdiff_t,
		  class Pointer = T*,
		  class Reference = T&
>
class random_access_iterator_base : public std::iterator<std::random_access_iterator_tag, T, Distance, Pointer, Reference>
{

public:

	using iterator_base = std::iterator<std::random_access_iterator_tag, T, Distance, Pointer, Reference>;
	using iterator_category = typename iterator_base::iterator_category;
	using value_type = typename iterator_base::value_type;
	using difference_type = typename iterator_base::difference_type;
	using pointer = typename iterator_base::pointer;
	using reference = typename iterator_base::reference;

	using derived_iterator = Derived;

	derived_iterator operator++(int)
	{
		derived_iterator& d(down_cast());
		derived_iterator tmp(d);
		++d;
		return tmp;
	}

	friend derived_iterator operator++(derived_iterator& d, int)
	{
		derived_iterator tmp(d);
		++d;
		return tmp;
	}

	derived_iterator operator--(int)
	{
		derived_iterator& d(down_cast());
		derived_iterator tmp(d);
		--d;
		return tmp;
	}

	friend derived_iterator operator+(const derived_iterator& d, difference_type n)
	{
		derived_iterator tmp(d);
		return tmp += n;
	}

	friend derived_iterator operator+(difference_type n, const derived_iterator& d)
	{
		return d + n;
	}

	friend derived_iterator operator-(const derived_iterator& d, difference_type n)
	{
		derived_iterator tmp(d);
		return tmp -= n;
	}

	reference operator[](difference_type n) const
	{
		return *(down_cast() + n);
	}

	bool operator!=(const derived_iterator& rhs) const
	{
		return !(down_cast() == rhs);
	}

	bool operator<=(const derived_iterator& rhs) const
	{
		return !(rhs < down_cast());
	}

	bool operator>=(const derived_iterator& rhs) const
	{
		return !(down_cast <rhs);
	}

	bool operator>(const derived_iterator& rhs) const
	{
		return rhs < down_cast();
	}

private:

	derived_iterator& down_cast()
	{
		return *static_cast<derived_iterator*>(this);
	}

	const derived_iterator& down_cast() const
	{
		return *static_cast<const derived_iterator*>(this);
	}
};

template <bool B, class T, class F>
using conditional_t = typename std::conditional<B, T, F>::type;

template <class Array, bool is_const>
class np_array_iterator
	: public random_access_iterator_base<np_array_iterator<Array, is_const>,
										 typename Array::value_type,
										 typename Array::difference_type,
										 conditional_t<is_const,
													   typename Array::const_pointer,
													   typename Array::pointer>,
										 conditional_t<is_const,
													   typename Array::const_reference,
													   typename Array::reference>>
{

public:

	using iterator_base = random_access_iterator_base<np_array_iterator<Array, is_const>,
													  typename Array::value_type,
													  typename Array::difference_type,
													  conditional_t<is_const,
																	typename Array::const_pointer,
																	typename Array::pointer>,
													  conditional_t<is_const,
																	typename Array::const_reference,
																	typename Array::reference >>;

	using iterator_category = typename iterator_base::iterator_category;
	using value_type = typename iterator_base::value_type;
	using difference_type = typename iterator_base::difference_type;
	using pointer = typename iterator_base::pointer;
	using reference = typename iterator_base::reference;

	explicit np_array_iterator(pointer ptr = nullptr) : p_ptr(ptr) {}

	np_array_iterator& operator++()
	{
		++p_ptr;
		return *this;
	}

	np_array_iterator& operator--()
	{
		--p_ptr;
		return *this;
	}

	np_array_iterator& operator+=(difference_type n)
	{
		p_ptr += n;
		return *this;
	}

	np_array_iterator operator-=(difference_type n)
	{
		p_ptr -= n;
		return *this;
	}

	reference operator*() const
	{
		return *p_ptr;
	}

	pointer operator->() const
	{
		return p_ptr;
	}

	bool operator==(const np_array_iterator& rhs) const
	{
		return p_ptr == rhs.p_ptr;
	}

	bool operator<(const np_array_iterator& rhs) const
	{
		return p_ptr < rhs.p_ptr;
	}

private:

	pointer p_ptr;
};

template <class T>
struct np_array_base_type
{
	using traits = std::allocator_traits<std::allocator<T>>;

	using value_type = typename traits::value_type;
	using reference = T&;
	using const_reference = const T&;
	using pointer = typename traits::pointer;
	using const_pointer = typename traits::const_pointer;
	using size_type = typename traits::size_type;
	using difference_type = typename traits::difference_type;
};

template <class T>
class np_array
{

public:

	using wrappee_type = array_t<T>;
	using base_type = np_array_base_type<T>;
	using value_type = typename base_type::value_type;
	using reference = typename base_type::reference;
	using const_reference = typename base_type::const_reference;
	using pointer = typename base_type::pointer;
	using const_pointer = typename base_type::const_pointer;
	using size_type = typename base_type::size_type;
	using difference_type = typename base_type::difference_type;

	using iterator = np_array_iterator<np_array<T>, false>;
	using const_iterator = np_array_iterator<np_array<T>, true>;

	using reverse_iterator = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;

	np_array() noexcept: m_wrappee(), m_buffer_info() {}
	
	explicit np_array(size_type size)
		: m_wrappee(buffer_info(nullptr, sizeof(T), format_descriptor<T>::value(),
					s_ndim, std::vector<size_t>(s_ndim, size), s_strides))
	{
		update_buffer_info();
	}

	np_array(size_type size, const_reference val)
		: m_wrappee(buffer_info(nullptr, sizeof(T), format_descriptor<T>::value(),
			s_ndim, std::vector<size_t>(s_ndim, size), s_strides))
	{
		std::fill(get_buffer(), get_buffer() + size, val);
	}

	template <class Iterator>
	np_array(Iterator first, Iterator last)
		: np_array(size_type(std::distance(first, last)))
	{
		std::copy(first, last, get_buffer());
	}

	np_array(const wrappee_type& wrappee)
		: m_wrappee(wrappee)
	{
		update_buffer_info();
	}

	np_array(const np_array& rhs)
		: m_wrappee(rhs.m_wrappee)
	{
		update_buffer_info();
	}

	np_array& operator=(const np_array& rhs)
	{
		if (this != &rhs)
		{
			m_wrappee = rhs.m_wrappee;
			update_buffer_info();
		}
		return *this;
	}

	np_array& operator=(const wrappee_type& wrappee)
	{
		m_wrappee = wrappee;
		update_buffer_info();
		return *this;
	}

	np_array(np_array&& rhs)
		: m_wrappee(std::move(rhs.m_wrappee))
	{
		update_buffer_info();
	}

	np_array& operator=(np_array&& rhs)
	{
		if (this = !&rhs)
		{
			m_wrappee = std::move(rhs.m_wrappee);
			update_buffer_info();
		}
		return *this;
	}

	np_array& operator=(wrappee_type&& wrappee)
	{
		m_wrappee = std::move(wrappee);
		update_buffer_info();
		return *this;
	}

	wrappee_type get_wrappee() const
	{
		return m_wrappee;
	}

	bool empty() const { return size() == 0; }
	size_type size() const { return m_buffer_info.size; }

	void resize(size_type size) { resize_impl(size); }
	void resize(size_type size, const_reference value)
	{
		resize_impl(size);
		std::fill(get_buffer(), get_buffer() + size, value);
	}

	reference operator[](size_type i) { return get_buffer()[i]; }
	const_reference operator[](size_type i) const { return get_buffer()[i]; }

	reference front() { return get_buger()[0]; }
	const_reference front() const { return get_buffer()[0]; }

	reference back() { return get_buffer()[size() - 1]; }
	const_reference back() const { return get_buffer()[size() - 1]; }

	iterator begin() noexcept { return iterator(get_buffer()); }
	iterator end() noexcept { return iterator(get_buffer() + size()); }

	const_iterator begin() const noexcept { return const_iterator(get_buffer()); }
	const_iterator end() const noexcept { return const_iterator(get_buffer() + size()); }

	const_iterator cbegin() const noexcept { return begin();  }
	const_iterator cend() const noexcept { return end(); }

	reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
	reverse_iterator rend() noexcept { return reverse_iterator(begin()); }

	const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
	const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }

	const_reverse_iterator crbegin() const noexcept { return rbegin(); }
	const_reverse_iterator crend() const noexcept { return rend(); }

private:

	pointer get_buffer() const
	{
		return reinterpret_cast<pointer>(m_buffer_info.ptr);
	}

	void update_buffer_info()
	{
		// TODO: consider adding move operation to buffer_info
		m_buffer_info = m_wrappee.request();
	}

	void resize_impl(size_type size)
	{
		m_wrappee = std::move(wrappee_type(buffer_info(nullptr, sizeof(T), format_descriptor<T>::value(),
										   s_ndim, std::vector<size_t>(s_ndim, size), s_strides)));
		update_buffer_info();
	}

	wrappee_type m_wrappee;
	buffer_info m_buffer_info;

	static constexpr int s_ndim = 1;
	static const std::vector<size_t> s_strides;
};

template <class T>
const std::vector<size_t> np_array<T>::s_strides = { sizeof(T) };

NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

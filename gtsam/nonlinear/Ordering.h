/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation, 
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    Ordering.h
 * @author  Richard Roberts
 * @date    Sep 2, 2010
 */

#pragma once

#include <map>
#include <set>
#include <gtsam/nonlinear/Key.h>
#include <gtsam/inference/inference.h>

#include <boost/foreach.hpp>
#include <boost/assign/list_inserter.hpp>
#include <boost/pool/pool_alloc.hpp>

namespace gtsam {

/**
 * An ordering is a map from symbols (non-typed keys) to integer indices
 */
class Ordering {
protected:
  typedef boost::fast_pool_allocator<std::pair<const Symbol, Index> > Allocator;
  typedef std::map<Symbol, Index, std::less<Symbol>, Allocator> Map;
  Map order_;
  Index nVars_;

public:

  typedef boost::shared_ptr<Ordering> shared_ptr;

  typedef std::pair<const Symbol, Index> value_type;
  typedef Map::iterator iterator;
  typedef Map::const_iterator const_iterator;

  /// Default constructor for empty ordering
  Ordering() : nVars_(0) {}

  /// Construct from list, assigns order indices sequentially to list items.
  Ordering(const std::list<Symbol> & L) ;

  /** One greater than the maximum ordering index, i.e. including missing indices in the count.  See also size(). */
  Index nVars() const { return nVars_; }

  /** The actual number of variables in this ordering, i.e. not including missing indices in the count.  See also nVars(). */
  Index size() const { return order_.size(); }

  iterator begin() { return order_.begin(); } /**< Iterator in order of sorted symbols, not in elimination/index order! */
  const_iterator begin() const { return order_.begin(); } /**< Iterator in order of sorted symbols, not in elimination/index order! */
  iterator end() { return order_.end(); } /**< Iterator in order of sorted symbols, not in elimination/index order! */
  const_iterator end() const { return order_.end(); } /**< Iterator in order of sorted symbols, not in elimination/index order! */

  // access to integer indices

  Index& at(const Symbol& key) { return operator[](key); } ///< Synonym for operator[](const Symbol&)
  Index at(const Symbol& key) const { return operator[](key); } ///< Synonym for operator[](const Symbol&) const

  /** Assigns the ordering index of the requested \c key into \c index if the symbol
   * is present in the ordering, otherwise does not modify \c index.  The
   * return value indicates whether the symbol is in fact present in the
   * ordering.
   * @param key The key whose index you request
   * @param [out] index Reference into which to write the index of the requested key, if the key is present.
   * @return true if the key is present and \c index was modified, false otherwise.
   */
  bool tryAt(const Symbol& key, Index& index) const {
    const_iterator i = order_.find(key);
    if(i != order_.end()) {
      index = i->second;
      return true;
    } else
      return false;
  }

  /// Access the index for the requested key, throws std::out_of_range if the
  /// key is not present in the ordering (note that this differs from the
  /// behavior of std::map)
  Index& operator[](const Symbol& key) {
    iterator i=order_.find(key);
    if(i == order_.end())  throw std::out_of_range(std::string());
    else                   return i->second; }

  /// Access the index for the requested key, throws std::out_of_range if the
  /// key is not present in the ordering (note that this differs from the
  /// behavior of std::map)
  Index operator[](const Symbol& key) const {
    const_iterator i=order_.find(key);
    if(i == order_.end())  throw std::out_of_range(std::string());
    else                   return i->second; }

  /** Returns an iterator pointing to the symbol/index pair with the requested,
   * or the end iterator if it does not exist.
   *
   * @return An iterator pointing to the symbol/index pair with the requested,
   * or the end iterator if it does not exist.
   */
  iterator find(const Symbol& key) { return order_.find(key); }

  /** Returns an iterator pointing to the symbol/index pair with the requested,
   * or the end iterator if it does not exist.
   *
   * @return An iterator pointing to the symbol/index pair with the requested,
   * or the end iterator if it does not exist.
   */
  const_iterator find(const Symbol& key) const { return order_.find(key); }

  // adding symbols

  /**
   * Attempts to insert a symbol/order pair with same semantics as stl::Map::insert(),
   * i.e., returns a pair of iterator and success (false if already present)
   */
  std::pair<iterator,bool> tryInsert(const value_type& key_order) {
  	std::pair<iterator,bool> it_ok(order_.insert(key_order));
  	if(it_ok.second == true && key_order.second+1 > nVars_)
  		nVars_ = key_order.second+1;
  	return it_ok;
  }
  std::pair<iterator,bool> tryInsert(const Symbol& key, Index order) { return tryInsert(std::make_pair(key,order)); }

  /** Try insert, but will fail if the key is already present */
  iterator insert(const value_type& key_order) {
  	std::pair<iterator,bool> it_ok(tryInsert(key_order));
  	if(!it_ok.second)  throw std::invalid_argument(std::string());
  	else               return it_ok.first;
  }
  iterator insert(const Symbol& key, Index order) { return insert(std::make_pair(key,order)); }

  /// Test if the key exists in the ordering.
  bool exists(const Symbol& key) const { return order_.count(key); }

  /// Adds a new key to the ordering with an index of one greater than the current highest index.
  Index push_back(const Symbol& key) { return insert(std::make_pair(key, nVars_))->second; }

  /** Remove the last (last-ordered, not highest-sorting key) symbol/index pair
   * from the ordering (this version is \f$ O(n) \f$, use it when you do not
   * know the last-ordered key).
   *
   * If you already know the last-ordered symbol, call popback(const Symbol&)
   * that accepts this symbol as an argument.
   *
   * @return The symbol and index that were removed.
   */
  value_type pop_back();

  /** Remove the last-ordered symbol from the ordering (this version is
   * \f$ O(\log n) \f$, use it if you already know the last-ordered key).
   *
   * Throws std::invalid_argument if the requested key is not actually the
   * last-ordered.
   *
   * @return The index of the symbol that was removed.
   */
  Index pop_back(const Symbol& key);

  /**
   * += operator allows statements like 'ordering += x0,x1,x2,x3;', which are
   * very useful for unit tests.  This functionality is courtesy of
   * boost::assign.
   */
  inline boost::assign::list_inserter<boost::assign_detail::call_push_back<Ordering>, Symbol>
  operator+=(const Symbol& key) {
    return boost::assign::make_list_inserter(boost::assign_detail::call_push_back<Ordering>(*this))(key); }

  /**
   * Reorder the variables with a permutation.  This is typically used
   * internally, permuting an initial key-sorted ordering into a fill-reducing
   * ordering.
   */
  void permuteWithInverse(const Permutation& inversePermutation);

  /** print (from Testable) for testing and debugging */
  void print(const std::string& str = "Ordering:") const;

  /** equals (from Testable) for testing and debugging */
  bool equals(const Ordering& rhs, double tol = 0.0) const;

private:

	/** Serialization function */
	friend class boost::serialization::access;
	template<class ARCHIVE>
	void serialize(ARCHIVE & ar, const unsigned int version) {
		ar & BOOST_SERIALIZATION_NVP(order_);
		ar & BOOST_SERIALIZATION_NVP(nVars_);
	}
};

/**
 * @class Unordered
 * @brief a set of unordered indices
 */
class Unordered: public std::set<Index> {
public:
  /** Default constructor creates empty ordering */
  Unordered() { }

  /** Create from a single symbol */
  Unordered(Index key) { insert(key); }

  /** Copy constructor */
  Unordered(const std::set<Index>& keys_in) : std::set<Index>(keys_in) {}

  /** whether a key exists */
  bool exists(const Index& key) { return find(key) != end(); }

  // Testable
  void print(const std::string& s = "Unordered") const;
  bool equals(const Unordered &t, double tol=0) const;
};

}


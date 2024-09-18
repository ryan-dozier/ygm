// Copyright 2019-2021 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include "ygm/detail/assert.hpp"
#undef NDEBUG

#include <ygm/comm.hpp>
#include <ygm/detail/layout.hpp>
#include <ygm/detail/shm_exchange.hpp>

int main(int argc, char** argv) {
  shm::shm_exchange exchanger();
}
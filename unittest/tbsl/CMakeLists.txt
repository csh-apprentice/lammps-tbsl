cmake_minimum_required(VERSION 3.16)
project(tbsl_ionization_tests LANGUAGES CXX)

# Keep this focused suite offline: the general LAMMPS unit-test configuration
# downloads googletest and libyaml, neither of which is needed here.
set(ENABLE_TESTING OFF CACHE BOOL "General LAMMPS tests" FORCE)
set(LAMMPS_EXCEPTIONS ON CACHE BOOL "LAMMPS library exceptions" FORCE)
add_subdirectory(../../cmake lammps)

enable_testing()
add_executable(test_ionization_pair test_ionization_pair.cpp)
target_link_libraries(test_ionization_pair PRIVATE lammps)

# Build the author's guarded-restart positive control and the exploratory
# review probes against the same library. The complete review executable also
# contains intentionally failing cases, so it is not registered with CTest.
add_executable(tbsl_custom_restart_protocol review/custom_restart/protocol.cpp)
target_link_libraries(tbsl_custom_restart_protocol PRIVATE lammps)
add_executable(tbsl_review_probes review/probes.cpp)
target_link_libraries(tbsl_review_probes PRIVATE lammps)

add_test(NAME tbsl_ionization_1rank COMMAND test_ionization_pair)
set(tbsl_test_names tbsl_ionization_1rank)
if(BUILD_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  foreach(nprocs 2 4)
    add_test(NAME tbsl_ionization_${nprocs}ranks
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nprocs}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_ionization_pair> ${MPIEXEC_POSTFLAGS})
    list(APPEND tbsl_test_names tbsl_ionization_${nprocs}ranks)
  endforeach()
endif()
set_tests_properties(${tbsl_test_names} PROPERTIES
  ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 30)

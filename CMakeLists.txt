cmake_minimum_required(VERSION 3.5)
project(aca_lab3)
#set(CMAKE_BUILD_TYPE Release)

include_directories(include)

set(SOURCE_FILES main.c
        src/main_drivers.c include/main_drivers.h
        src/matmat.c include/matmat.h
        src/util.c include/util.h
        src/matvec.c include/matvec.h
        src/sse_methods.c include/sse_methods.h
        src/matvec_auto.c include/matvec_auto.h
        src/matmat_auto.c include/matmat_auto.h)

#set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-O3 -march=native -mtune=native -funroll-loops -ftree-vectorize -ftree-vectorizer-verbose=1  -ffast-math -fassociative-math")
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3 -march=native -mtune=native -funroll-loops -ftree-vectorize -ftree-vectorizer-verbose=1  -ffast-math -fassociative-math")
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-std=gnu99 -msse4.1 -msse3 -O2 -funroll-loops -Wall -Wextra -m64 -mtune=native")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=gnu99 -msse4.1 -msse3 -O2 -funroll-loops -Wall -Wextra -m64 -mtune=native")

#set_source_files_properties(src/matvec_auto.c PROPERTIES COMPILE_FLAGS "-fopenmp -m64 -mtune=native -O2 -unroll-loops -funroll-loops -ftree-vectorizer-verbose=4 -fopt-info-all -fopt-info-missed-all -ftree-loop-vectorize -fopt-info-vec-missed")
#set_source_files_properties(src/matmat_auto.c PROPERTIES COMPILE_FLAGS "-fopenmp -m64 -mtune=native -O2 -unroll-loops -funroll-loops -ftree-vectorizer-verbose=4 -fopt-info-all -fopt-info-missed-all -ftree-loop-vectorize -fopt-info-vec-missed")
set_source_files_properties(src/matmat_auto.c PROPERTIES COMPILE_FLAGS "-m64 -mtune=native -O2 -funroll-loops -unroll-loops -ftree-vectorizer-verbose=2 -fopt-info-vec-all -ftree-loop-vectorize -fopt-info-vec-missed")
set_source_files_properties(src/matvec_auto.c PROPERTIES COMPILE_FLAGS "-m64 -mtune=native -O2 -funroll-loops -unroll-loops -ftree-vectorizer-verbose=2 -fopt-info-vec-all -ftree-loop-vectorize -fopt-info-vec-missed")
add_executable(aca_lab3 ${SOURCE_FILES})
target_link_libraries(aca_lab3 m)

include(CTest)
enable_testing()
# Does application run
add_test(aca_lab3Dorun aca_lab3 -h)
set_tests_properties(aca_lab3Dorun PROPERTIES PASS_REGULAR_EXPRESSION "Show this menu")

add_test(aca_lab3{mt4} aca_lab3 -n 4 -mt)
set_tests_properties(aca_lab3{mt4} PROPERTIES PASS_REGULAR_EXPRESSION "(7 SSE mxm.*verified.*listing 7 C.*- OK).*(7 auto.*verified.*listing 7 C.*- OK)")

add_test(aca_lab3vt4 aca_lab3 -n 4 -vt)
set_tests_properties(aca_lab3vt4 PROPERTIES PASS_REGULAR_EXPRESSION "(SSE.*version verified against listing 5 C Program - OK)\
.*(Listing 6 C Program verified against listing 5 C Program - OK)\
.*(Auto vectorized version verified against simple version - OK)")
set(CTEST_PARALLEL_LEVEL 3)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()
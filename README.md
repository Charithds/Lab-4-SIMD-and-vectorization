# aca_lab3
SSE Implementations

###Where each algorithm implemented
|      Algorithm              | Location                                             |
|-----------------------------|----------------------------------------------------|
|C program of Listing 5       | src/matvec.c *:matvec_simple_listing5*                        |
|C program of Listing 6       | src/matvec.c *:matvec_unrolled_listing6*                        |
|SSE version of of Listing 6  | src/ssemethods.c *:matvec_unrolled_16sse*                       |
|C program of Listing 7       | src/matmat.c *:matmat_listing7*                        |
|SSE version of listing 7     | src/ssemethods.c *:matmat_listing7_sse*                        |
|Auto vectorized ver Listing 7| src/matmat_auto.c *:matmat_auto*                        |

###Command Line arguments [not complete]
| argument | effect                                             |
|----------|----------------------------------------------------|
| -n N     | will change dimensions of the **square matrices      |
| -s       | will run the SSE version                           |

* ** In matrix matrix multiplication test second matrix column size is hardcoded as 200,
  change `#define COLUMNS 200` in main.c to change it
* make sure all the dimensions are **multiple of 4**, since we have assumed it on loop unrolling and
    no checking is done in release build at runtime to avoid program crash due to bad inputs

###Building the Project
Create a folder inside the project. `eg : build`. Then within the folder
run `cmake ..`, In linux this will usually create a Makefile to compile the project. 
You can run `make` within the folder to build the project and `make test` to run the tests.

### Testing
We have included 3 test to check whether program runs and we verify
the SSE and Auto vectorized versions against serial algorithm. To invoke these 3 tests
in parallel run the below command after building the program.

`> make test ARGS=-j3` 
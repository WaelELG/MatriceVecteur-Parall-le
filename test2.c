#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define BLOCK_LOW(id,p,n) ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n) ((id+1)*(n)/(p) - 1)
#define BLOCK_SIZE(id,p,n) ((id+1)*(n)/(p) - (id)*(n)/(p))
#define BLOCK_OWNER(index,p,n) (((p)*((index)+1)-1)/(n))

void **matrix_create(size_t m, size_t n, size_t size) {
    size_t i; 
    void **p= (void **) malloc(m*n*size+ m*sizeof(void *));
    char *c=  (char*) (p+m);
    for(i=0; i<m; ++i)
        p[i]= (void *) c+i*n*size;
    return p;
}

void matrix_print(double **M, size_t m, size_t n, char *name) {
    size_t i,j;
    printf("%s=[",name);
    for(i=0; i<m; ++i) {
        printf("\n  ");
        for(j=0; j<n; ++j)
            printf("%f  ",M[i][j]);
    }
    printf("\n];\n");
}

main(int argc, char *argv[]) {

    int npes, myrank, root = 0, n = 7, rows, i, j, *sendcounts, *displs;
    double **m, **mParts;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&npes);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    // Matrix M is generated in the root process (process 0)
    if (myrank == root) {
        m = (double**)matrix_create(n, n, sizeof(double));
        for (i = 0; i < n; ++i)
            for (j = 0; j < n; ++j)
                m[i][j] = (double)(n * i + j);
    }

    // Array containing the numbers of rows for each process
    sendcounts = malloc(n * sizeof(int));
    // Array containing the displacement for each data chunk
    displs = malloc(n * sizeof(int));
    // For each process ...
    for (j = 0; j < npes; j++) {
        // Sets each number of rows
        sendcounts[j] = BLOCK_SIZE(j, npes, n)*n;
        // Sets each displacement
        displs[j] = BLOCK_LOW(j, npes, n)*n;
    }
    // Each process gets the number of rows that he is going to get
    rows = sendcounts[myrank]/n;
    // Creates the empty matrixes for the parts of M
    mParts = (double**)matrix_create(rows, n, sizeof(double));
    // Scatters the matrix parts through all the processes
    MPI_Scatterv(&m[0][0], sendcounts, displs, MPI_DOUBLE, &mParts[0][0], sendcounts[myrank], MPI_DOUBLE, root, MPI_COMM_WORLD);

    // This is where I get the Segmentation Fault
    if (myrank == 1) matrix_print(mParts, rows, n, "mParts");

    MPI_Finalize();
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define USE_MPI
#ifdef USE_MPI
#include <mpi.h>
#else
#include "mpiDummy.h"
#endif

#define PI 3.1415926535897932384626433832795

/* Pure phi-field grain-growth test.
 * No concentration, no chemical potential, no KKS, no extrapolation.
 */
#define Nx 200
#define Nz 200
#define DT 0.0007

#define NUM_BETA 10
#define NUM_GAMMA 10
#define NUM_COMP (1 + NUM_BETA + NUM_GAMMA + 1)

#define X_BETA_START  80
#define X_BETA_END    100
#define X_GAMMA_START 100
#define X_GAMMA_END   120

#define S 0.8
#define PERTURB_AMP 6.0

#define PHI_CUTOFF 1e-14
#define ACTIVE_EPS 1e-10
#define DPHI_LIMIT 0.02

#define M_ALL 1.0
#define W_ALL 1.0
#define A2_ALL 2.0
#define TIME_SCALE 1.0

#define OUT_EVERY 100
#define OUT_FIRST_STEPS 10
#define MAX_STEPS 20000

#define MPI_MASTER 0

enum { IDX_AL=0, IDX_MG=1, IDX_BETA=2, IDX_GAMMA=3 };

static double *phi[NUM_COMP+2];
static double *phi_t[NUM_COMP+2];
static double *lap_phi[NUM_COMP+2];
static double *comm_buf;
static double *comm_buf1;

#define PHI(n,x,z)   phi[n][(x)*(Nz+2)+(z)]
#define PHIT(n,x,z)  phi_t[n][((x)-1)*Nz+((z)-1)]
#define LAP(n,x,z)   lap_phi[n][(x)*(Nz+2)+(z)]

static int get_type(int phase_id)
{
    if(phase_id == 1) return IDX_AL;
    if(phase_id == NUM_COMP) return IDX_MG;
    if(phase_id >= 2 && phase_id <= 1 + NUM_BETA) return IDX_BETA;
    return IDX_GAMMA;
}

static void die_if(int cond, const char *msg)
{
    if(cond) {
        fprintf(stderr, "%s\n", msg);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static int xdir_mpi_sr(int mpi_rank, int mpi_size)
{
    int blocksize = Nx / mpi_size;
    int buf_len = Nz + 2;
    int total_size = NUM_COMP * buf_len;
    int left = (mpi_rank == 0) ? MPI_PROC_NULL : mpi_rank - 1;
    int right = (mpi_rank == mpi_size - 1) ? MPI_PROC_NULL : mpi_rank + 1;

    if(left != MPI_PROC_NULL) {
        double *sbuf = comm_buf;
        double *rbuf = comm_buf + total_size;
        for(int n=1; n<=NUM_COMP; n++)
            memcpy(sbuf + (n-1)*buf_len, &PHI(n,1,0), buf_len*sizeof(double));
        MPI_Sendrecv(sbuf, total_size, MPI_DOUBLE, left, 0,
                     rbuf, total_size, MPI_DOUBLE, left, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int n=1; n<=NUM_COMP; n++)
            memcpy(&PHI(n,0,0), rbuf + (n-1)*buf_len, buf_len*sizeof(double));
    }

    if(right != MPI_PROC_NULL) {
        double *sbuf = comm_buf1;
        double *rbuf = comm_buf1 + total_size;
        for(int n=1; n<=NUM_COMP; n++)
            memcpy(sbuf + (n-1)*buf_len, &PHI(n,blocksize,0), buf_len*sizeof(double));
        MPI_Sendrecv(sbuf, total_size, MPI_DOUBLE, right, 0,
                     rbuf, total_size, MPI_DOUBLE, right, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int n=1; n<=NUM_COMP; n++)
            memcpy(&PHI(n,blocksize+1,0), rbuf + (n-1)*buf_len, buf_len*sizeof(double));
    }

    return 0;
}

static int set_bc(int mpi_rank, int mpi_size)
{
    int blocksize = Nx / mpi_size;
    for(int i=0; i<=blocksize+1; i++) {
        for(int n=1; n<=NUM_COMP; n++) {
            PHI(n,i,0) = PHI(n,i,Nz);
            PHI(n,i,Nz+1) = PHI(n,i,1);
        }
    }
    if(mpi_rank == 0) {
        for(int k=0; k<=Nz+1; k++)
            for(int n=1; n<=NUM_COMP; n++)
                PHI(n,0,k) = PHI(n,2,k);
    }
    if(mpi_rank == mpi_size - 1) {
        for(int k=0; k<=Nz+1; k++)
            for(int n=1; n<=NUM_COMP; n++)
                PHI(n,blocksize+1,k) = PHI(n,blocksize-1,k);
    }
    return 0;
}

static int local_phase_present(int n, int i, int k)
{
    if(PHI(n,i,k) > ACTIVE_EPS) return 1;
    if(fabs(LAP(n,i,k)) > ACTIVE_EPS) return 1;
    return 0;
}

static int set_init(int mpi_rank, int mpi_size)
{
    int blocksize = Nx / mpi_size;
    double beta_bounds[NUM_BETA+1];
    double gamma_bounds[NUM_GAMMA+1];

    if(mpi_rank == 0) {
        srand(1);
        double sum_h = 0.0;
        double raw_b[NUM_BETA];
        for(int n=0; n<NUM_BETA; n++) {
            raw_b[n] = 12.0 + 16.0 * (double)rand() / (double)RAND_MAX;
            sum_h += raw_b[n];
        }
        beta_bounds[0] = 0.0;
        for(int n=0; n<NUM_BETA; n++)
            beta_bounds[n+1] = beta_bounds[n] + raw_b[n] * (double)Nz / sum_h;
        beta_bounds[NUM_BETA] = (double)Nz;

        double sum_g = 0.0;
        double raw_g[NUM_GAMMA];
        for(int n=0; n<NUM_GAMMA; n++) {
            raw_g[n] = 12.0 + 16.0 * (double)rand() / (double)RAND_MAX;
            sum_g += raw_g[n];
        }
        gamma_bounds[0] = 0.0;
        for(int n=0; n<NUM_GAMMA; n++)
            gamma_bounds[n+1] = gamma_bounds[n] + raw_g[n] * (double)Nz / sum_g;
        gamma_bounds[NUM_GAMMA] = (double)Nz;
    }

    MPI_Bcast(beta_bounds, NUM_BETA+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(gamma_bounds, NUM_GAMMA+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for(int i=0; i<=blocksize+1; i++) {
        for(int k=0; k<=Nz+1; k++) {
            double gx = (double)(mpi_rank * blocksize + i);
            double gz = (double)(k - 1);
            double theta = 2.0 * PI * gz / (double)Nz;
            double x_ab = (double)X_BETA_START + PERTURB_AMP * cos(theta);
            double x_gm = (double)X_GAMMA_END - PERTURB_AMP * cos(theta);

            for(int n=1; n<=NUM_COMP; n++) PHI(n,i,k) = 0.0;

            PHI(1,i,k) = 0.5 * (1.0 - tanh(S * (gx - x_ab)));

            double phi_x_beta = 0.5 * (tanh(S * (gx - x_ab))
                                     - tanh(S * (gx - X_BETA_END)));
            if(phi_x_beta < 0.0) phi_x_beta = 0.0;
            for(int n=0; n<NUM_BETA; n++) {
                double zc = 0.5 * (beta_bounds[n] + beta_bounds[n+1]);
                double zw = 0.5 * (beta_bounds[n+1] - beta_bounds[n]);
                double dz = fabs(gz - zc);
                if(dz > 0.5 * (double)Nz) dz = (double)Nz - dz;
                double phi_z = 0.5 * (1.0 - tanh(S * (dz - zw)));
                if(phi_z < 0.0) phi_z = 0.0;
                PHI(2+n,i,k) = phi_x_beta * phi_z;
            }

            double phi_x_gamma = 0.5 * (tanh(S * (gx - X_GAMMA_START))
                                      - tanh(S * (gx - x_gm)));
            if(phi_x_gamma < 0.0) phi_x_gamma = 0.0;
            for(int n=0; n<NUM_GAMMA; n++) {
                double zc = 0.5 * (gamma_bounds[n] + gamma_bounds[n+1]);
                double zw = 0.5 * (gamma_bounds[n+1] - gamma_bounds[n]);
                double dz = fabs(gz - zc);
                if(dz > 0.5 * (double)Nz) dz = (double)Nz - dz;
                double phi_z = 0.5 * (1.0 - tanh(S * (dz - zw)));
                if(phi_z < 0.0) phi_z = 0.0;
                PHI(2+NUM_BETA+n,i,k) = phi_x_gamma * phi_z;
            }

            PHI(NUM_COMP,i,k) = 0.5 * (1.0 + tanh(S * (gx - x_gm)));

            double sum = 0.0;
            for(int n=1; n<=NUM_COMP; n++) {
                if(PHI(n,i,k) < PHI_CUTOFF) PHI(n,i,k) = 0.0;
                sum += PHI(n,i,k);
            }
            if(sum > 1e-14) {
                double inv = 1.0 / sum;
                for(int n=1; n<=NUM_COMP; n++) PHI(n,i,k) *= inv;
            } else {
                PHI(1,i,k) = 1.0;
            }
        }
    }

    return 0;
}

static int time_step(double h, int mpi_rank, int mpi_size)
{
    int blocksize = Nx / mpi_size;
    double h2 = h * h;

    set_bc(mpi_rank, mpi_size);
    xdir_mpi_sr(mpi_rank, mpi_size);
    set_bc(mpi_rank, mpi_size);

    for(int i=1; i<=blocksize; i++) {
        for(int k=1; k<=Nz; k++) {
            for(int n=1; n<=NUM_COMP; n++) {
                LAP(n,i,k) = (PHI(n,i+1,k) + PHI(n,i-1,k)
                            + PHI(n,i,k+1) + PHI(n,i,k-1)
                            - 4.0 * PHI(n,i,k)) / h2;
            }
        }
    }

    for(int i=1; i<=blocksize; i++) {
        for(int k=1; k<=Nz; k++) {
            int present[NUM_COMP+2];
            int n_present = 0;

            for(int n=1; n<=NUM_COMP; n++) {
                present[n] = local_phase_present(n,i,k);
                if(present[n]) n_present++;
            }

            for(int n=1; n<=NUM_COMP; n++) {
                if(!present[n]) {
                    PHIT(n,i,k) = 0.0;
                    continue;
                }

                double rhs = 0.0;
                for(int j=1; j<=NUM_COMP; j++) {
                    if(j == n || !present[j]) continue;

                    double phase_term = W_ALL * (PHI(n,i,k) - PHI(j,i,k));
                    double grad_term = 0.5 * A2_ALL * (LAP(n,i,k) - LAP(j,i,k));
                    rhs += 2.0 * M_ALL * (phase_term + grad_term);
                }
                PHIT(n,i,k) = (n_present > 0) ? rhs / (double)n_present : 0.0;
            }
        }
    }

    for(int i=1; i<=blocksize; i++) {
        for(int k=1; k<=Nz; k++) {
            double sum = 0.0;
            for(int n=1; n<=NUM_COMP; n++) {
                double dphi = TIME_SCALE * DT * PHIT(n,i,k);
                if(dphi > DPHI_LIMIT) dphi = DPHI_LIMIT;
                if(dphi < -DPHI_LIMIT) dphi = -DPHI_LIMIT;
                double p = PHI(n,i,k) + dphi;
                if(p < PHI_CUTOFF) p = 0.0;
                if(p > 1.0) p = 1.0;
                PHI(n,i,k) = p;
                sum += p;
            }
            if(sum > 1e-14) {
                double inv = 1.0 / sum;
                for(int n=1; n<=NUM_COMP; n++) PHI(n,i,k) *= inv;
            } else {
                PHI(1,i,k) = 1.0;
                for(int n=2; n<=NUM_COMP; n++) PHI(n,i,k) = 0.0;
            }
        }
    }

    return 0;
}

static int write_scalar_vtk(const char *folder, const char *name, int step,
                            double *local_buf, int blocksize, int mpi_rank,
                            int mpi_size, int tag)
{
    int local_size = blocksize * Nz;
    int global_size = Nx * Nz;
    double *global_buf = NULL;
    char filename[512];

    if(mpi_rank == 0) {
        global_buf = (double*)malloc(global_size * sizeof(double));
        if(!global_buf) return -1;
        memcpy(global_buf, local_buf, local_size*sizeof(double));
        for(int r=1; r<mpi_size; r++)
            MPI_Recv(global_buf + r*local_size, local_size, MPI_DOUBLE,
                     r, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        snprintf(filename, sizeof(filename), "%s/%s_step%d.vtk", folder, name, step);
        FILE *fp = fopen(filename, "w");
        if(fp) {
            fprintf(fp, "# vtk DataFile Version 3.0\n%s step %d\nASCII\n", name, step);
            fprintf(fp, "DATASET STRUCTURED_POINTS\n");
            fprintf(fp, "DIMENSIONS %d %d 1\n", Nx, Nz);
            fprintf(fp, "ORIGIN 0 0 0\nSPACING 1.0 1.0 1.0\n");
            fprintf(fp, "POINT_DATA %d\n", global_size);
            fprintf(fp, "SCALARS %s float 1\nLOOKUP_TABLE default\n", name);
            for(int z=0; z<Nz; z++)
                for(int x=0; x<Nx; x++)
                    fprintf(fp, "%g\n", global_buf[x*Nz + z]);
            fclose(fp);
        }
        free(global_buf);
    } else {
        MPI_Send(local_buf, local_size, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    }
    return 0;
}

static int epsout(const char *folder, int step, int mpi_rank, int mpi_size)
{
    int blocksize = Nx / mpi_size;
    int local_size = blocksize * Nz;
    double *buf = (double*)malloc(local_size * sizeof(double));
    if(!buf) return -1;

    if(mpi_rank == 0) mkdir(folder, 0755);
    MPI_Barrier(MPI_COMM_WORLD);

    int idx = 0;
    for(int i=1; i<=blocksize; i++) {
        for(int k=1; k<=Nz; k++) {
            int dom = 1;
            double vmax = PHI(1,i,k);
            for(int n=2; n<=NUM_COMP; n++) {
                if(PHI(n,i,k) > vmax) {
                    vmax = PHI(n,i,k);
                    dom = n;
                }
            }
            int cls = get_type(dom);
            if(cls == IDX_AL) buf[idx++] = 0.0;
            else if(cls == IDX_BETA) buf[idx++] = 1.0;
            else if(cls == IDX_GAMMA) buf[idx++] = 2.0;
            else buf[idx++] = 3.0;
        }
    }
    write_scalar_vtk(folder, "phase_class", step, buf, blocksize, mpi_rank, mpi_size, 100);

    idx = 0;
    for(int i=1; i<=blocksize; i++) {
        for(int k=1; k<=Nz; k++) {
            double vmax = 0.0;
            for(int n=1; n<=NUM_COMP; n++)
                if(PHI(n,i,k) > vmax) vmax = PHI(n,i,k);
            buf[idx++] = 1.0 - vmax;
        }
    }
    write_scalar_vtk(folder, "interface", step, buf, blocksize, mpi_rank, mpi_size, 101);

    free(buf);
    if(mpi_rank == 0) printf("output step %d\n", step);
    return 0;
}

int main(int argc, char **argv)
{
    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    die_if(Nx % mpi_size != 0, "Nx must be divisible by mpi size");
    int blocksize = Nx / mpi_size;
    int ghost_size = (blocksize + 2) * (Nz + 2);
    int phys_size = blocksize * Nz;
    (void)phys_size;

    for(int n=1; n<=NUM_COMP; n++) {
        phi[n] = (double*)calloc(ghost_size, sizeof(double));
        phi_t[n] = (double*)calloc(phys_size, sizeof(double));
        lap_phi[n] = (double*)calloc(ghost_size, sizeof(double));
        die_if(!phi[n] || !phi_t[n] || !lap_phi[n], "allocation failed");
    }

    int comm_size = NUM_COMP * (Nz + 2);
    comm_buf = (double*)calloc(2 * comm_size, sizeof(double));
    comm_buf1 = (double*)calloc(2 * comm_size, sizeof(double));
    die_if(!comm_buf || !comm_buf1, "communication buffer allocation failed");

    set_init(mpi_rank, mpi_size);
    set_bc(mpi_rank, mpi_size);
    xdir_mpi_sr(mpi_rank, mpi_size);
    set_bc(mpi_rank, mpi_size);

    const char *folder = "phi_1_output";
    epsout(folder, 0, mpi_rank, mpi_size);

    for(int step=1; step<=MAX_STEPS; step++) {
        time_step(1.0, mpi_rank, mpi_size);
        set_bc(mpi_rank, mpi_size);
        xdir_mpi_sr(mpi_rank, mpi_size);
        set_bc(mpi_rank, mpi_size);

        if(step <= OUT_FIRST_STEPS || step % OUT_EVERY == 0)
            epsout(folder, step, mpi_rank, mpi_size);
    }

    for(int n=1; n<=NUM_COMP; n++) {
        free(phi[n]);
        free(phi_t[n]);
        free(lap_phi[n]);
    }
    free(comm_buf);
    free(comm_buf1);

    MPI_Finalize();
    return 0;
}

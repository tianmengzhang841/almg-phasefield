#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "malloc.h"
#include "memory.h"
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#define USE_MPI
#ifdef USE_MPI
#include <mpi.h>
#else
#include "mpiDummy.h"
#endif

#define PI 3.1415926535897932384626433832795

/* ======================================================
 * 网格参数（加密）
 * ====================================================== */
#define Nx  200
#define Nz  200
#define DT  0.0007   /* 网格加密后时间步需相应缩小 */

/* ======================================================
 * 相场布局
 *   相1      : Al
 *   相2      : β
 *   相3      : γ
 *   相4      : Mg
 *   NUM_COMP = 4
 *
 *   X方向:
 *     Al   : x ∈ [0,  80)   → 80格 (40%)
 *     β区  : x ∈ [80, 100)  → 20格 (10%)
 *     γ区  : x ∈ [100,120)  → 20格 (10%)
 *     Mg   : x ∈ [120,200)  → 80格 (40%)
 *
 *   Z方向（中间相）:
 *     当前测试使用较稀疏的 beta/gamma 多晶粒，
 *     避免晶粒高度与界面宽度过于接近。
 * ====================================================== */
#define NUM_BETA   10
#define NUM_GAMMA  10
#define NUM_COMP   (1 + NUM_BETA + NUM_GAMMA + 1)   /* = 22 */

#define X_AL_END      80
#define X_BETA_START  80
#define X_BETA_END    100
#define X_GAMMA_START 100
#define X_GAMMA_END   120
#define X_MG_START    120

/* tanh 界面锐度（界面宽约 2/S 格，S=0.8时宽约2.5格，适合小晶粒）*/
#define S  0.8

/* 外侧 Al/Beta 和 Gamma/Mg 曲线界面的振幅（格点数） */
#define PERTURB_AMP  6.0

/* 被吞并判断阈值：全场最大phi < 此值则标记该相为已消亡
 * 只对中间相（Beta/Gamma）判断，Al和Mg永远活跃
 * 阈值设为 1e-6，避免界面处正常的小phi值被误判 */
#define EXTINCT_THRESH  1e-6

/* 平衡浓度（Mg摩尔分数）*/
#define C_AL    0.15
#define C_BETA  0.39
#define C_GAMMA 0.55
#define C_MG    0.85

/* 热力学参数 */
#define MAX_ITER_C_INV  50
#define TOL_C           1e-10
#define EPSILON_PHI     1e-6
#define PHI_SMOOTH_LOW  1e-8
#define PHI_SMOOTH_HIGH 1e-4
#define PHI_ZERO_CUTOFF 1e-12    /* numerical cleanup only; physical presence is still phi > 0 */
#define C_MOBILITY      10.0    /* concentration diffusion/relaxation mobility */

/* Interface mobilities used in phi evolution. */
#define M_SAME_IMC      3.0
#define M_BETA_GAMMA    3.0
#define M_AL_BETA       3.0
#define M_GAMMA_MG      3.0
#define M_OTHER_IMC     1.0
#define CHEM_DRIVE_SCALE (0.03)     /* match chemical and phi-interface driving-force magnitudes */

/* Concentration mobility multipliers used inside the diffusion flux. */
#define D_AL            1.0
#define D_BETA          5.0
#define D_GAMMA         5.0
#define D_MG            1.0
#define D_BETA_GAMMA_GB 0.00
#define EXTRAP_AVG_COLS 3
#define CI_Z_FILTER_ALPHA 0.20
#define DEBUG_AB_KKS 0
#define DEBUG_EVERY_STEP_UNTIL 10

#define TIMELIMIT  10000

/* 晶粒几何参数 */
#define N_GRAINS     5       /* 每侧晶粒数 */
#define R_LARGE      70.0   /* 大晶粒半径（格点）*/
#define R_SMALL      30.0   /* 小晶粒半径（格点）*/
#define X_INTERFACE  100.0  /* β/γ界面 x 坐标 */
 
/* 各晶粒圆心Z坐标和半径（外凸/内凹标志）
 * convex=1: 向左凸（β看到凸面），convex=0: 向右凸（β看到凹面）*/
static const double grain_zc[N_GRAINS]     = {70.0, 170.0, 270.0, 370.0, 440.0};
static const double grain_R[N_GRAINS]      = {70.0,  30.0,  70.0,  30.0,  40.0};
static const int    grain_convex[N_GRAINS] = {  1,     0,     1,     0,     1 };

#define MPI_MASTER    0
#define STATE_EXIT    0
#define STATE_RUN     1
#define STATE_CONTINUE 2
#define STATE_OUT_MSG  5
#define STATE_OUT_EPS  6

/* 热力学相索引 */
#define IDX_AL    0
#define IDX_MG    1
#define IDX_BETA  2
#define IDX_GAMMA 3

/* 在 mu_field 声明之后添加 */
double* diag_ci_al;    /* Al局部浓度诊断场 */
double* diag_ci_beta;  /* Beta族加权平均局部浓度 */
double* diag_ci_gamma; /* Gamma族加权平均局部浓度 */
double* diag_ci_mg;    /* Mg局部浓度诊断场 */

double T_temp = 400.0;

/* ======================================================
 * 全局相活跃标志（1=活跃，0=已消亡）
 * 在每次输出时更新，time_step 中跳过消亡相
 * ====================================================== */
int phase_active[NUM_COMP+2];  /* 1-indexed */

/* ======================================================
 * 热力学函数
 * ====================================================== */
int get_thermo_type(int phase_id) {
    if (phase_id == 1)                            return IDX_AL;
    if (phase_id == NUM_COMP)                     return IDX_MG;
    if (phase_id >= 2 && phase_id <= 1+NUM_BETA)  return IDX_BETA;
    return IDX_GAMMA;
}

double f_Al(double c) {
    double T=T_temp, c2=c*c, c3=c2*c, c4=c3*c;
    return (-0.1686*T-46.2846)+(-34.5737*T-122.7264)*c
          +1e3*(0.0664*T+5.7068)*c2+1e3*(-0.0664*T-8.5771)*c3
          +1e3*(0.0335*T+5.3517)*c4;
}
double f_Mg(double c) {
    double T=T_temp, c2=c*c, c3=c2*c, c4=c3*c;
    return 1e3*(-0.002*T+4.9527)+1e4*(-0.0029*T-1.3340)*c
          +1e4*(0.0064*T+1.7468)*c2+1e4*(-0.0067*T-1.8225)*c3
          +1e3*(0.0334*T+9.1053)*c4;
}
double f_Beta(double c) {
    double T=T_temp, off=c-0.386;
    return 1e3*(-0.003*T-1.9197)+1e5*off*off;
}
double f_Gamma(double c) {
    double T=T_temp, c2=c*c, c3=c2*c, c4=c3*c;
    return 12.5714*T+475.5374+1e4*(-0.0160*T+3.4198)*c
          +1e5*(0.0050*T-1.8963)*c2+1e5*(-0.0062*T+2.8921)*c3
          +1e5*(0.0027*T-1.3086)*c4;
}

double mu_Al(double c) {
    double T=T_temp;
    return (-34.5737*T-122.7264)+2e3*(0.0664*T+5.7068)*c
          +3e3*(-0.0664*T-8.5771)*c*c+4e3*(0.0335*T+5.3517)*c*c*c;
}
double mu_Mg(double c) {
    double T=T_temp;
    return 1e4*(-0.0029*T-1.3340)+2e4*(0.0064*T+1.7468)*c
          +3e4*(-0.0067*T-1.8225)*c*c+4e3*(0.0334*T+9.1053)*c*c*c;
}
double mu_Beta(double c)  { return 2e5*(c-0.386); }
double mu_Gamma(double c) {
    double T=T_temp;
    return 1e4*(-0.0160*T+3.4198)+2e5*(0.0050*T-1.8963)*c
          +3e5*(-0.0062*T+2.8921)*c*c+4e5*(0.0027*T-1.3086)*c*c*c;
}

double get_mu(int phase_id, double c) {
    int t=get_thermo_type(phase_id);
    if(t==IDX_AL)   return mu_Al(c);
    if(t==IDX_MG)   return mu_Mg(c);
    if(t==IDX_BETA) return mu_Beta(c);
    return mu_Gamma(c);
}

/* ======================================================
 * 全局场指针
 * ====================================================== */
double* phi[NUM_COMP+2];
double* ci[NUM_COMP+2];
double* lap_phi[NUM_COMP+2];
double* chem_pot[NUM_COMP+2];
double* phi_t[NUM_COMP+2];
double* c_total;
double* mu_field;
double* graphic_x;   /* 从 main 局部变量改为全局 */
double* graphic_z;   /* 从 main 局部变量改为全局 */
char debug_folder1[256] = ".";

#define PHI(n,x,z)    phi[n][(x)*(Nz+2)+(z)]
#define CI(n,x,z)     ci[n][(x)*(Nz+2)+(z)]
#define PHI_t(n,x,z)  phi_t[n][((x)-1)*(Nz)+((z)-1)]
#define LAP(n,x,z)    lap_phi[n][(x)*(Nz+2)+(z)]
#define C_total(x,z)  c_total[(x)*(Nz+2)+(z)]
#define C_t(x,z)      c_t[((x)-1)*(Nz)+((z)-1)]
#define GRAPHICX(x,z) graphic_x[(x)*(Nz+2)+(z)]
#define GRAPHICZ(x,z) graphic_z[(x)*(Nz+2)+(z)]

/* ======================================================
 * 函数声明
 * ====================================================== */
int set_init(double **phi,double *c_total,double **ci,double h,int mpi_rank,int mpi_size);
int set_bc(double **phi,double *c_total,double **ci,double h,int mpi_rank,int mpi_size);
int time_step(double **phi,double **ci,double **lap_phi,double **chem_pot,double **phi_t,
              double *c_total,double *c_t,double *graphic_x,double *graphic_z,int **phi_real_field,
              double *comm_buf,double *comm_buf1,double *f3,double h,double alpha,
              int mpi_rank,int mpi_size,unsigned long iloop);
int xdir_mpi_sr(double **phi,double **ci,double *c_total,double *graphic_x,double *graphic_z,
                int **phi_real_field,double *comm_buf,double *comm_buf1,
                int mpi_rank,int mpi_size);
int epsout(double **phi,double *c_total,double **ci,double *c_t,double h,int mpi_rank,int mpi_size,int eps_tag,char *folder1);
int energy_epsout(double **phi,double **ci,double h,int mpi_rank,int mpi_size,int eps_tag,char *folder1);
int msgout(double **phi,double *c_total,double **ci,double *f3,double h,int mpi_rank,int mpi_size,double msg_time,unsigned long iloop);
int read_field(double **phi,double **ci,double *c_total,double *comm_buf,double *comm_buf1,char *folder,int index,int mpi_rank,int mpi_size);
void update_active_phases(double **phi,int mpi_rank,int mpi_size);
void solve_local_equilibrium(double *phi_vals,double C_total_val,double *c_eq,double *mu_star,int *phi_real_flag);
double invert_mu_to_c(int phase_id,double mu_target);
void get_concentration_bounds(int phase_id, double *c_low, double *c_high);
double clamp_concentration(int phase_id, double c);
double smooth_phi_weight(double phi_val);
double concentration_diffusivity(double phi_al, double phi_beta,
                                 double phi_gamma, double phi_mg);

void fill_nearest_valid_scalar(double *local_value, int *local_valid, double *local_out,
                               int blocksize, int mpi_rank)
{
    int local_phys_size = blocksize * Nz;
    int global_phys_size = Nx * Nz;
    double *global_value = (double*)malloc(global_phys_size * sizeof(double));
    int *global_valid = (int*)malloc(global_phys_size * sizeof(int));

    if(!global_value || !global_valid) {
        for(int q=0; q<local_phys_size; q++) local_out[q] = local_value[q];
        free(global_value); free(global_valid);
        return;
    }

    MPI_Allgather(local_value, local_phys_size, MPI_DOUBLE,
                  global_value, local_phys_size, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(local_valid, local_phys_size, MPI_INT,
                  global_valid, local_phys_size, MPI_INT, MPI_COMM_WORLD);

    for(int lx=1; lx<=blocksize; lx++) {
        int gx = mpi_rank * blocksize + (lx - 1);
        for(int z=0; z<Nz; z++) {
            int local_idx = (lx - 1) * Nz + z;
            int global_idx = gx * Nz + z;
            if(global_valid[global_idx]) {
                local_out[local_idx] = global_value[global_idx];
                continue;
            }

            int best = -1;
            for(int d=1; d<Nx; d++) {
                int xl = gx - d;
                int xr = gx + d;
                if(xl >= 0 && global_valid[xl*Nz + z]) { best = xl; break; }
                if(xr < Nx && global_valid[xr*Nz + z]) { best = xr; break; }
            }
            local_out[local_idx] = (best >= 0) ? global_value[best*Nz + z]
                                               : local_value[local_idx];
        }
    }

    free(global_value); free(global_valid);
}

double concentration_diffusivity(double phi_al, double phi_beta,
                                 double phi_gamma, double phi_mg)
{
    if(phi_al >= 0.8) return D_AL;
    if(phi_mg >= 0.8) return D_MG;
    if(phi_al > 0.2 && phi_al < 0.8) {
        return (D_AL > D_BETA) ? D_AL : D_BETA;
    }
    if(phi_mg > 0.2 && phi_mg < 0.8) {
        return (D_MG > D_GAMMA) ? D_MG : D_GAMMA;
    }
    if(phi_beta >= 0.8) return D_BETA;
    if(phi_gamma >= 0.8) return D_GAMMA;
    return D_BETA_GAMMA_GB;
}

double smooth_phi_weight(double phi_val)
{
    if(phi_val <= PHI_SMOOTH_LOW) return 0.0;
    if(phi_val >= PHI_SMOOTH_HIGH) return phi_val;
    double s = (phi_val - PHI_SMOOTH_LOW) / (PHI_SMOOTH_HIGH - PHI_SMOOTH_LOW);
    s = s * s * (3.0 - 2.0 * s);
    return phi_val * s;
}

void fill_side_average_scalar(int type, double *local_value, int *local_valid,
                              double *local_out, int blocksize, int mpi_rank)
{
    int local_phys_size = blocksize * Nz;
    int global_phys_size = Nx * Nz;
    double *global_value = (double*)malloc(global_phys_size * sizeof(double));
    int *global_valid = (int*)malloc(global_phys_size * sizeof(int));

    if(!global_value || !global_valid) {
        for(int q=0; q<local_phys_size; q++) local_out[q] = local_value[q];
        free(global_value); free(global_valid);
        return;
    }

    MPI_Allgather(local_value, local_phys_size, MPI_DOUBLE,
                  global_value, local_phys_size, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(local_valid, local_phys_size, MPI_INT,
                  global_valid, local_phys_size, MPI_INT, MPI_COMM_WORLD);

    int global_min_x = Nx;
    int global_max_x = -1;
    for(int gx=0; gx<Nx; gx++) {
        for(int z=0; z<Nz; z++) {
            if(global_valid[gx*Nz + z]) {
                if(gx < global_min_x) global_min_x = gx;
                if(gx > global_max_x) global_max_x = gx;
            }
        }
    }

    if(global_max_x < global_min_x) {
        double fallback = (type == IDX_AL) ? C_AL :
                          (type == IDX_BETA) ? C_BETA :
                          (type == IDX_GAMMA) ? C_GAMMA : C_MG;
        for(int q=0; q<local_phys_size; q++) local_out[q] = fallback;
        free(global_value); free(global_valid);
        return;
    }

    double mid_x = 0.5 * (global_min_x + global_max_x);
    double stats[4] = {0.0, 0.0, 0.0, 0.0};

    for(int gx=0; gx<Nx; gx++) {
        for(int z=0; z<Nz; z++) {
            int gidx = gx * Nz + z;
            if(!global_valid[gidx]) continue;
            int boundary = 0;
            if(gx == 0 || !global_valid[(gx-1)*Nz + z]) boundary = 1;
            if(gx == Nx-1 || !global_valid[(gx+1)*Nz + z]) boundary = 1;
            if(z == 0 || !global_valid[gx*Nz + (z-1)]) boundary = 1;
            if(z == Nz-1 || !global_valid[gx*Nz + (z+1)]) boundary = 1;
            if(!boundary) continue;

            if((double)gx <= mid_x) {
                stats[0] += global_value[gidx];
                stats[2] += 1.0;
            } else {
                stats[1] += global_value[gidx];
                stats[3] += 1.0;
            }
        }
    }

    double left_avg  = (stats[2] > 0.0) ? stats[0] / stats[2] : 0.0;
    double right_avg = (stats[3] > 0.0) ? stats[1] / stats[3] : left_avg;
    if(stats[2] <= 0.0) left_avg = right_avg;

    for(int lx=1; lx<=blocksize; lx++) {
        int gx = mpi_rank * blocksize + (lx - 1);
        for(int z=0; z<Nz; z++) {
            int idx = (lx - 1) * Nz + z;
            if(local_valid[idx]) {
                local_out[idx] = local_value[idx];
            } else {
                local_out[idx] = ((double)gx <= mid_x) ? left_avg : right_avg;
            }
        }
    }

    free(global_value); free(global_valid);
}

void fill_type_ci_buffer(int type, double *local_out, int blocksize, int mpi_rank, int mask_absent)
{
    int local_phys_size = blocksize * Nz;
    double *raw = (double*)calloc(local_phys_size, sizeof(double));
    int *valid = (int*)calloc(local_phys_size, sizeof(int));
    if(!raw || !valid) {
        free(raw); free(valid);
        return;
    }

    for(int x=1; x<=blocksize; x++) {
        for(int zz=1; zz<=Nz; zz++) {
            int idx = (x-1)*Nz + (zz-1);
            if(type == IDX_AL) {
                valid[idx] = (PHI(1,x,zz) > 0.0);
                raw[idx] = CI(1,x,zz);
            } else if(type == IDX_MG) {
                valid[idx] = (PHI(NUM_COMP,x,zz) > 0.0);
                raw[idx] = CI(NUM_COMP,x,zz);
            } else {
                int first = (type == IDX_BETA) ? 2 : 2 + NUM_BETA;
                int last  = (type == IDX_BETA) ? 1 + NUM_BETA : 1 + NUM_BETA + NUM_GAMMA;
                double sum_phi = 0.0;
                double sum_ci = 0.0;
                for(int n=first; n<=last; n++) {
                    if(!phase_active[n]) continue;
                    double p = PHI(n,x,zz);
                    if(p > 0.0) {
                        sum_phi += p;
                        sum_ci += p * CI(n,x,zz);
                    }
                }
                if(sum_phi > 0.0) {
                    valid[idx] = 1;
                    raw[idx] = sum_ci / sum_phi;
                } else {
                    raw[idx] = (type == IDX_BETA) ? C_BETA : C_GAMMA;
                }
            }
        }
    }
    if(mask_absent) {
        for(int q=0; q<local_phys_size; q++)
            local_out[q] = valid[q] ? raw[q] : NAN;
    } else {
        fill_side_average_scalar(type, raw, valid, local_out, blocksize, mpi_rank);
    }
    free(raw); free(valid);
}

void filter_ci_z_after_extrapolation(int blocksize)
{
    if(CI_Z_FILTER_ALPHA <= 0.0) return;
    double alpha = CI_Z_FILTER_ALPHA;
    if(alpha > 0.5) alpha = 0.5;

    double *line = (double*)malloc((Nz + 2) * sizeof(double));
    if(!line) return;

    for(int n=1; n<=NUM_COMP; n++) {
        if(!phase_active[n]) continue;
        for(int i=1; i<=blocksize; i++) {
            for(int k=1; k<=Nz; k++)
                line[k] = CI(n,i,k);
            for(int k=1; k<=Nz; k++) {
                double c_left  = (k == 1)  ? line[2]    : line[k-1];
                double c_right = (k == Nz) ? line[Nz-1] : line[k+1];
                double c_new = (1.0 - alpha) * line[k]
                             + 0.5 * alpha * (c_left + c_right);
                CI(n,i,k) = clamp_concentration(n, c_new);
            }
        }
    }

    free(line);
}

double family_local_ci(int type, int x, int z);
void write_ab_kks_debug(const char *stage, unsigned long step, char *folder1,
                        int blocksize, int mpi_rank);
void extrapolate_ceq_zero_phi(int **phi_real_field, int blocksize, int mpi_rank, int mpi_size);
void fill_nearest_valid_scalar(double *local_value, int *local_valid, double *local_out,
                               int blocksize, int mpi_rank);
void fill_side_average_scalar(int type, double *local_value, int *local_valid,
                              double *local_out, int blocksize, int mpi_rank);
void fill_type_ci_buffer(int type, double *local_out, int blocksize, int mpi_rank, int mask_absent);
void filter_ci_z_after_extrapolation(int blocksize);
void refresh_local_ci_from_current_phi(int **phi_real_field, double *comm_buf,
                                       double *comm_buf1, int mpi_rank, int mpi_size);


void refresh_local_ci_from_current_phi(int **phi_real_field, double *comm_buf,
                                       double *comm_buf1, int mpi_rank, int mpi_size)
{
    int blocksize = Nx / mpi_size;
    for(int i=1; i<=blocksize; i++) {
        for(int k=1; k<=Nz; k++) {
            double phi_now[NUM_COMP+2];
            double c_eq_now[NUM_COMP+2];
            double mu_now;
            int flag_now[NUM_COMP+2];
            for(int n=1; n<=NUM_COMP; n++)
                phi_now[n] = phase_active[n] ? PHI(n,i,k) : 0.0;
            solve_local_equilibrium(phi_now, C_total(i,k), c_eq_now, &mu_now, flag_now);
            for(int n=1; n<=NUM_COMP; n++) {
                phi_real_field[n][i*(Nz+2)+k] = flag_now[n];
                if(phase_active[n]) CI(n,i,k) = c_eq_now[n];
            }
        }
    }

    for(int i=0; i<=blocksize+1; i++) {
        for(int n=1; n<=NUM_COMP; n++) {
            if(!phase_active[n]) continue;
            CI(n,i,0)    = CI(n,i,1);
            CI(n,i,Nz+1) = CI(n,i,Nz);
        }
    }
    xdir_mpi_sr(phi, ci, c_total, graphic_x, graphic_z,
                phi_real_field, comm_buf, comm_buf1, mpi_rank, mpi_size);
    extrapolate_ceq_zero_phi(phi_real_field, blocksize, mpi_rank, mpi_size);
    filter_ci_z_after_extrapolation(blocksize);
    for(int i=0; i<=blocksize+1; i++) {
        for(int n=1; n<=NUM_COMP; n++) {
            if(!phase_active[n]) continue;
            CI(n,i,0)    = CI(n,i,1);
            CI(n,i,Nz+1) = CI(n,i,Nz);
        }
    }
    xdir_mpi_sr(phi, ci, c_total, graphic_x, graphic_z,
                phi_real_field, comm_buf, comm_buf1, mpi_rank, mpi_size);
}

/* ======================================================
 * main
 * ====================================================== */
int main(int argc, char **argv)
{
    double alpha;
    double *comm_buf, *comm_buf1, *f3;
    double dx;
    int state, ret, i, n;
    unsigned long iloop, imsg, ieps;
    int mpi_rank, mpi_size, blocksize, oldindex;
    double start_time, t, msg_time, eps_time;
    char namebuf1[80];
    double *c_t;
    FILE *file1;

    dx = 0.4;
    msg_time = 20;
    eps_time = 20;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    ret=0; state=STATE_EXIT; oldindex=0; file1=NULL;
    blocksize = Nx / mpi_size;
    if (Nx % mpi_size != 0) {
        if(mpi_rank==0) fprintf(stderr,"Error: Nx not divisible by mpi_size\n");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    /* 初始化所有相为活跃 */
    for (n=1; n<=NUM_COMP; n++) phase_active[n] = 1;

    int nx_alloc    = blocksize+2;
    int nz_alloc    = Nz+2;
    int size_ghost  = nx_alloc * nz_alloc;
    int phys_size   = blocksize * Nz;

    for (i=1; i<=NUM_COMP; ++i) {
        phi[i]      = (double*)calloc(size_ghost, sizeof(double));
        ci[i]       = (double*)calloc(size_ghost, sizeof(double));
        lap_phi[i]  = (double*)calloc(size_ghost, sizeof(double));
        chem_pot[i] = (double*)calloc(size_ghost, sizeof(double));
        phi_t[i]    = (double*)calloc(phys_size,  sizeof(double));
        if(!phi[i]||!ci[i]||!lap_phi[i]||!chem_pot[i]||!phi_t[i]) {
            fprintf(stderr,"Rank %d: alloc failed comp %d\n",mpi_rank,i);
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }
    int *phi_real_field[NUM_COMP+2];
    for(n=1; n<=NUM_COMP; n++)
    phi_real_field[n] = (int*)calloc((blocksize+2)*(Nz+2), sizeof(int));

    c_total   = (double*)calloc(size_ghost, sizeof(double));
    mu_field  = (double*)calloc(size_ghost, sizeof(double));
    graphic_x = (double*)calloc(size_ghost, sizeof(double));
    graphic_z = (double*)calloc(size_ghost, sizeof(double));
    c_t       = (double*)calloc(phys_size,  sizeof(double));
    f3        = (double*)calloc(phys_size,  sizeof(double));


    /* 在 f3 = calloc 之后添加 */
    diag_ci_al    = (double*)calloc(phys_size, sizeof(double));
    diag_ci_beta  = (double*)calloc(phys_size, sizeof(double));
    diag_ci_gamma = (double*)calloc(phys_size, sizeof(double));
    diag_ci_mg    = (double*)calloc(phys_size, sizeof(double));

    //int total_buf_size = (NUM_COMP + 1 + NUM_COMP) * (Nz+2);
    int total_buf_size = (NUM_COMP + 1 + NUM_COMP + 1 + 1) * (Nz+2);
    comm_buf  = (double*)malloc(sizeof(double)*2*total_buf_size);
    comm_buf1 = (double*)malloc(sizeof(double)*2*total_buf_size);
    if(!comm_buf||!comm_buf1) { MPI_Abort(MPI_COMM_WORLD,1); }

    alpha = 6.0;
    char folder1[64]={0};
    snprintf(folder1,sizeof(folder1),"A=%.6f",alpha);
    snprintf(debug_folder1, sizeof(debug_folder1), "%s", folder1);
    mkdir(folder1,0777);

    state = STATE_EXIT;
    start_time = MPI_Wtime();
    t = start_time;

    if (mpi_rank==MPI_MASTER) {
        file1=fopen("phase.999.dat","r");
        if(file1!=NULL) { state=STATE_CONTINUE; fclose(file1); }
    }
    MPI_Bcast(&state,1,MPI_INT,MPI_MASTER,MPI_COMM_WORLD);

    if (state!=STATE_CONTINUE) {
        if(mpi_rank==MPI_MASTER) printf("Starting new run...\n");
        set_init(phi,c_total,ci,dx,mpi_rank,mpi_size);
    } else {
        if(mpi_rank==MPI_MASTER) printf("Resuming from step 999...\n");
        ret=read_field(phi,ci,c_total,comm_buf,comm_buf1,folder1,999,mpi_rank,mpi_size);
        set_bc(phi,c_total,ci,dx,mpi_rank,mpi_size);
        if(mpi_rank==MPI_MASTER) {
            if(ret!=-1) system("rm -f phase.999.dat");
            for(i=2;i<999;i++) {
                sprintf(namebuf1,"phase.%d.dat",i);
                file1=fopen(namebuf1,"r");
                if(file1==NULL) oldindex=i-1;
                if(file1!=NULL) fclose(file1);
                if(oldindex==(i-1)) break;
            }
        }
    }
    state = (ret!=-1) ? STATE_RUN : STATE_EXIT;
    MPI_Bcast(&state,1,MPI_INT,MPI_MASTER,MPI_COMM_WORLD);

    iloop=0; imsg=1; ieps=1;
    do {
        if(iloop==0) {
            /* 预更新CI为局部平衡值，保证初始输出语义正确 */
    int bs = blocksize;
    for(i=1; i<=bs; i++) {
        for(int kk=1; kk<=Nz; kk++) {
            double phi_pre[NUM_COMP+2];
            double c_eq_pre[NUM_COMP+2];
            double mu_pre;
            for(n=1; n<=NUM_COMP; n++)
                phi_pre[n] = phase_active[n] ? PHI(n,i,kk) : 0.0;
            int flag_pre[NUM_COMP+2];
            solve_local_equilibrium(phi_pre, C_total(i,kk), c_eq_pre, &mu_pre, flag_pre);
            for(n=1; n<=NUM_COMP; n++) {
                phi_real_field[n][i*(Nz+2)+kk] = flag_pre[n];
                if(!phase_active[n]) continue;
                CI(n,i,kk) = c_eq_pre[n];
            }
        }
    }
    /* Z方向ghost层 */
    for(i=0; i<=bs+1; i++) {
        for(n=1; n<=NUM_COMP; n++) {
            if(!phase_active[n]) continue;
            CI(n,i,0)    = CI(n,i,1);
            CI(n,i,Nz+1) = CI(n,i,Nz);
        }
    }
    xdir_mpi_sr(phi,ci,c_total,graphic_x,graphic_z,phi_real_field,comm_buf,comm_buf1,mpi_rank,mpi_size);
    extrapolate_ceq_zero_phi(phi_real_field, bs, mpi_rank, mpi_size);
    filter_ci_z_after_extrapolation(bs);
    for(i=0; i<=bs+1; i++) {
        for(n=1; n<=NUM_COMP; n++) {
            if(!phase_active[n]) continue;
            CI(n,i,0)    = CI(n,i,1);
            CI(n,i,Nz+1) = CI(n,i,Nz);
        }
    }
    xdir_mpi_sr(phi,ci,c_total,graphic_x,graphic_z,phi_real_field,comm_buf,comm_buf1,mpi_rank,mpi_size);
    set_bc(phi,c_total,ci,dx,mpi_rank,mpi_size);
    fill_type_ci_buffer(IDX_AL, diag_ci_al, bs, mpi_rank, 0);
    fill_type_ci_buffer(IDX_BETA, diag_ci_beta, bs, mpi_rank, 0);
    fill_type_ci_buffer(IDX_GAMMA, diag_ci_gamma, bs, mpi_rank, 0);
    fill_type_ci_buffer(IDX_MG, diag_ci_mg, bs, mpi_rank, 0);

            epsout(phi,c_total,ci,c_t,dx,mpi_rank,mpi_size,0,folder1);
            //energy_epsout(phi,ci,dx,mpi_rank,mpi_size,0,folder1);
        }
       
        xdir_mpi_sr(phi,ci,c_total,graphic_x,graphic_z,NULL,comm_buf,comm_buf1,mpi_rank,mpi_size);  /* [MOD-2] */
        set_bc(phi,c_total,ci,dx,mpi_rank,mpi_size);                        /* [MOD-2] */
        update_active_phases(phi,mpi_rank,mpi_size);
        time_step(phi,ci,lap_phi,chem_pot,phi_t,c_total,c_t,graphic_x,graphic_z,phi_real_field,comm_buf,comm_buf1,f3,dx,alpha,mpi_rank,mpi_size,iloop);
        xdir_mpi_sr(phi,ci,c_total,graphic_x,graphic_z,NULL,comm_buf,comm_buf1,mpi_rank,mpi_size);
        set_bc(phi,c_total,ci,dx,mpi_rank,mpi_size);

        if(iloop < DEBUG_EVERY_STEP_UNTIL || (iloop + 1) % 100 == 0) {
            epsout(phi,c_total,ci,c_t,dx,mpi_rank,mpi_size,(int)(iloop+1),folder1);
        }

        if(iloop%100==0) {
            if(mpi_rank==MPI_MASTER) {
                t=MPI_Wtime();
                if(t-start_time>msg_time*imsg && t-start_time<=eps_time*ieps) { state=STATE_OUT_MSG; imsg++; }
                if(t-start_time>eps_time*ieps) { state=STATE_OUT_EPS; ieps++; imsg++; }
                if(t-start_time>TIMELIMIT) state=STATE_EXIT;
            }
            MPI_Bcast(&state,1,MPI_INT,MPI_MASTER,MPI_COMM_WORLD);
            if(state==STATE_OUT_MSG) {
                msgout(phi,c_total,ci,f3,dx,mpi_rank,mpi_size,t-start_time,iloop);
                state=STATE_RUN;
            }
            if(state==STATE_OUT_EPS) {
                unsigned long out_step = iloop + 1;
                msgout(phi,c_total,ci,f3,dx,mpi_rank,mpi_size,t-start_time,out_step);
                
                /* VTK output is controlled by the fixed step schedule above.
                 * Keep this time-triggered branch for progress messages only,
                 * otherwise step 100 also produces a stale step1 file. */
                //epsout(phi,c_total,ci,c_t,dx,mpi_rank,mpi_size,ieps-1+oldindex,folder1);
                //energy_epsout(phi,ci,dx,mpi_rank,mpi_size,ieps-1+oldindex,folder1);
                state=STATE_RUN;
            }
            MPI_Bcast(&state,1,MPI_INT,MPI_MASTER,MPI_COMM_WORLD);
        }
        iloop++;
    } while(state==STATE_RUN);

    update_active_phases(phi,mpi_rank,mpi_size);
    msgout(phi,c_total,ci,f3,dx,mpi_rank,mpi_size,999,iloop);
    epsout(phi,c_total,ci,c_t,dx,mpi_rank,mpi_size,999,folder1);
    //energy_epsout(phi,ci,dx,mpi_rank,mpi_size,999,folder1);

    MPI_Finalize();

    for(i=1;i<=NUM_COMP;++i) {
        free(phi[i]); free(ci[i]); free(lap_phi[i]); free(chem_pot[i]); free(phi_t[i]);
    }
    free(c_total); free(mu_field); free(c_t);
    free(graphic_x); free(graphic_z); free(f3);
    free(comm_buf); free(comm_buf1);
    for(n=1; n<=NUM_COMP; n++) free(phi_real_field[n]);
    free(diag_ci_al); free(diag_ci_beta);
    free(diag_ci_gamma); free(diag_ci_mg);
    return 0;
}

/* ======================================================
 * update_active_phases
 * 计算每个相在全场的最大 phi，若 < EXTINCT_THRESH 则标记消亡
 * ====================================================== */
void update_active_phases(double **phi, int mpi_rank, int mpi_size)
{
    int blocksize = Nx/mpi_size;
    int n, i, k;
    double local_max[NUM_COMP+2];
    double global_max[NUM_COMP+2];

    for(n=1;n<=NUM_COMP;n++) {
        local_max[n]=0.0;
        for(i=1;i<=blocksize;i++)
            for(k=1;k<=Nz;k++)
                if(PHI(n,i,k)>local_max[n]) local_max[n]=PHI(n,i,k);
    }
    for(n=1;n<=NUM_COMP;n++)
        MPI_Allreduce(&local_max[n],&global_max[n],1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

    int newly_dead=0;
    for(n=1;n<=NUM_COMP;n++) {
        /* Al(1) 和 Mg(NUM_COMP) 永远活跃，不参与消亡判断 */
        if(n==1 || n==NUM_COMP) continue;
        if(global_max[n] < EXTINCT_THRESH && phase_active[n]==1) {
            phase_active[n]=0;
            newly_dead++;
            if(mpi_rank==0)
                printf("[Extinct] Phase %d eliminated (max_phi=%.2e)\n",n,global_max[n]);
        }
    }
    /* 消亡相的 phi 全部清零 */
    if(newly_dead>0) {
        for(n=1;n<=NUM_COMP;n++) {
            if(!phase_active[n]) {
                int sz=(blocksize+2)*(Nz+2);
                memset(phi[n],0,sz*sizeof(double));
                memset(ci[n], 0,sz*sizeof(double));
            }
        }
    }
}

/* ======================================================
 * set_init
 *
 * X方向: Al | β | γ | Mg
 * beta/gamma 在 z 方向分别划分为 NUM_BETA/NUM_GAMMA 个随机高度晶粒。
 * Al/Beta 和 Gamma/Mg 外侧界面为余弦曲线，Beta/Gamma 中间界面保持直线。
 * ====================================================== */

#define MIN_GRAIN_H  12.0   /* minimum grain height in grid points */
#define MAX_GRAIN_H  28.0   /* maximum grain height in grid points */
int set_init(double **phi, double *c_total, double **ci,
             double h, int mpi_rank, int mpi_size)
{
    int i, k, n;
    int blocksize = Nx / mpi_size;
 
    /* ---- Step 1: generate random beta/gamma grain boundaries on rank0 ---- */
    double beta_bounds[NUM_BETA+1];
    double gamma_bounds[NUM_GAMMA+1];
 
    if(mpi_rank == 0) {
        double raw_h[NUM_BETA], sum_h = 0.0;
        for(n=0; n<NUM_BETA; n++) {
            raw_h[n] = MIN_GRAIN_H
                     + (MAX_GRAIN_H - MIN_GRAIN_H) * (double)rand() / (double)RAND_MAX;
            sum_h += raw_h[n];
        }
        double scale = (double)Nz / sum_h;
        beta_bounds[0] = 0.0;
        for(n=0; n<NUM_BETA; n++)
            beta_bounds[n+1] = beta_bounds[n] + raw_h[n] * scale;
        beta_bounds[NUM_BETA] = (double)Nz;

        double raw_hg[NUM_GAMMA], sum_hg = 0.0;
        for(n=0; n<NUM_GAMMA; n++) {
            raw_hg[n] = MIN_GRAIN_H
                      + (MAX_GRAIN_H - MIN_GRAIN_H) * (double)rand() / (double)RAND_MAX;
            sum_hg += raw_hg[n];
        }
        double scaleg = (double)Nz / sum_hg;
        gamma_bounds[0] = 0.0;
        for(n=0; n<NUM_GAMMA; n++)
            gamma_bounds[n+1] = gamma_bounds[n] + raw_hg[n] * scaleg;
        gamma_bounds[NUM_GAMMA] = (double)Nz;
 
        printf("[set_init] multigrain curvature test: NUM_BETA=%d, NUM_GAMMA=%d, curve_amp=%.2f\n",
               NUM_BETA, NUM_GAMMA, PERTURB_AMP);
        printf("[set_init] Beta  grain heights: ");
        for(n=0; n<NUM_BETA;  n++) printf("%.1f ", beta_bounds[n+1] - beta_bounds[n]);
        printf("\n");
        printf("[set_init] Gamma grain heights: ");
        for(n=0; n<NUM_GAMMA; n++) printf("%.1f ", gamma_bounds[n+1] - gamma_bounds[n]);
        printf("\n");
    }
 
    /* broadcast to all ranks */
    MPI_Bcast(beta_bounds,  NUM_BETA+1,  MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(gamma_bounds, NUM_GAMMA+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 
    /* ---- Step 2: fill phase fields ---- */
    for(i=0; i<=blocksize+1; i++) {
        for(k=0; k<Nz+2; k++) {
 
            double gx = (double)(i + blocksize*mpi_rank);
            double gz = (double)(k - 1);
            double theta = 2.0 * PI * gz / (double)Nz;
            double x_al_beta = (double)X_BETA_START + PERTURB_AMP * cos(theta);
            double x_gamma_mg = (double)X_GAMMA_END   - PERTURB_AMP * cos(theta);
 
            for(n=1; n<=NUM_COMP; n++) { PHI(n,i,k)=0.0; CI(n,i,k)=0.0; }
            C_total(i,k) = 0.0;
 
            /* --- Al --- */
            double phi_Al = 0.5*(1.0 - tanh(S*(gx - x_al_beta)));
            PHI(1,i,k)    = phi_Al;
            CI(1,i,k)     = C_AL * phi_Al;
            C_total(i,k) += C_AL * phi_Al;
 
            /* --- beta grain: left boundary is curved, beta/gamma boundary is straight --- */
            double phi_x_beta = 0.5*(tanh(S*(gx - x_al_beta))
                                   - tanh(S*(gx - X_BETA_END)));
            if(phi_x_beta < 0.0) phi_x_beta = 0.0;
 
            for(n=0; n<NUM_BETA; n++) {
                double z_lo  = beta_bounds[n];
                double z_hi  = beta_bounds[n+1];
                double z_center = 0.5 * (z_lo + z_hi);
                double z_half_width = 0.5 * (z_hi - z_lo);
                double z_dist = fabs(gz - z_center);
                if(z_dist > 0.5 * (double)Nz) z_dist = (double)Nz - z_dist;
                double phi_z = 0.5*(1.0 - tanh(S*(z_dist - z_half_width)));
                if(phi_z < 0.0) phi_z = 0.0;
                double p  = phi_x_beta * phi_z;
                int    pn = 2 + n;              /* phase index 2..NUM_BETA+1 */
                PHI(pn,i,k)   = p;
                CI(pn,i,k)    = C_BETA * p;
                C_total(i,k) += C_BETA * p;
            }
 
            /* --- gamma grain: beta/gamma boundary is straight, right boundary is curved --- */
            double phi_x_gamma = 0.5*(tanh(S*(gx - X_GAMMA_START))
                                    - tanh(S*(gx - x_gamma_mg)));
            if(phi_x_gamma < 0.0) phi_x_gamma = 0.0;
 
            for(n=0; n<NUM_GAMMA; n++) {
                double z_lo  = gamma_bounds[n];
                double z_hi  = gamma_bounds[n+1];
                double z_center = 0.5 * (z_lo + z_hi);
                double z_half_width = 0.5 * (z_hi - z_lo);
                double z_dist = fabs(gz - z_center);
                if(z_dist > 0.5 * (double)Nz) z_dist = (double)Nz - z_dist;
                double phi_z = 0.5*(1.0 - tanh(S*(z_dist - z_half_width)));
                if(phi_z < 0.0) phi_z = 0.0;
                double p  = phi_x_gamma * phi_z;
                int    pn = 2 + NUM_BETA + n;   /* phase index NUM_BETA+2..NUM_COMP-1 */
                PHI(pn,i,k)   = p;
                CI(pn,i,k)    = C_GAMMA * p;
                C_total(i,k) += C_GAMMA * p;
            }
 
            /* --- Mg --- */
            double phi_Mg = 0.5*(1.0 + tanh(S*(gx - x_gamma_mg)));
            PHI(NUM_COMP,i,k)  = phi_Mg;
            CI(NUM_COMP,i,k)   = C_MG * phi_Mg;
            C_total(i,k)      += C_MG * phi_Mg;
 
            /* --- normalize: ensure sum(phi) = 1 --- */
            double phi_sum = 0.0;
            for(n=1; n<=NUM_COMP; n++) phi_sum += PHI(n,i,k);
            if(phi_sum > 1e-12) {
                double inv = 1.0 / phi_sum;
                for(n=1; n<=NUM_COMP; n++) {
                    PHI(n,i,k) *= inv;
                    CI(n,i,k)  *= inv;
                }
                C_total(i,k) *= inv;
            } else {
                PHI(1,i,k)  = 1.0;
                CI(1,i,k)   = C_AL;
                C_total(i,k)= C_AL;
            }

            phi_sum = 0.0;
            for(n=1; n<=NUM_COMP; n++) {
                if(PHI(n,i,k) < PHI_ZERO_CUTOFF) PHI(n,i,k) = 0.0;
                phi_sum += PHI(n,i,k);
            }
            if(phi_sum > 1e-12) {
                double inv = 1.0 / phi_sum;
                C_total(i,k) = 0.0;
                for(n=1; n<=NUM_COMP; n++) {
                    PHI(n,i,k) *= inv;
                    double c_phase;
                    int tt = get_thermo_type(n);
                    if(tt == IDX_AL) c_phase = C_AL;
                    else if(tt == IDX_BETA) c_phase = C_BETA;
                    else if(tt == IDX_GAMMA) c_phase = C_GAMMA;
                    else c_phase = C_MG;
                    CI(n,i,k) = c_phase * PHI(n,i,k);
                    C_total(i,k) += CI(n,i,k);
                }
            }
        }
    }
    return 0;
}

/* ======================================================
 * concentration bounds / invert_mu_to_c
 * ====================================================== */
void get_concentration_bounds(int phase_id, double *c_low, double *c_high) {
    int type = get_thermo_type(phase_id);
    if(type == IDX_AL) {
        *c_low = 1e-9;
        *c_high = C_BETA;
    } else if(type == IDX_BETA) {
        *c_low = C_AL;
        *c_high = C_GAMMA;
    } else if(type == IDX_GAMMA) {
        *c_low = C_BETA;
        *c_high = C_MG;
    } else {
        *c_low = C_GAMMA;
        *c_high = 1.0 - 1e-9;
    }
}

double clamp_concentration(int phase_id, double c) {
    double c_low, c_high;
    get_concentration_bounds(phase_id, &c_low, &c_high);
    if(c < c_low) return c_low;
    if(c > c_high) return c_high;
    return c;
}

double invert_mu_to_c(int phase_id, double mu_target) {
    int type = get_thermo_type(phase_id);
    double c_low, c_high;
    get_concentration_bounds(phase_id, &c_low, &c_high);

    if(type == IDX_BETA)
        return clamp_concentration(phase_id, mu_target / 2e5 + 0.386);

    double mu_low = get_mu(phase_id, c_low);
    double mu_high = get_mu(phase_id, c_high);
    if(mu_low <= mu_high) {
        if(mu_target <= mu_low) return c_low;
        if(mu_target >= mu_high) return c_high;
    } else {
        if(mu_target >= mu_low) return c_low;
        if(mu_target <= mu_high) return c_high;
    }

    for(int iter = 0; iter < MAX_ITER_C_INV; iter++) {
        double c_mid = 0.5 * (c_low + c_high);
        double mu_mid = get_mu(phase_id, c_mid);
        if(fabs(mu_mid - mu_target) < TOL_C || (c_high - c_low) < 1e-12)
            return c_mid;
        if((mu_low <= mu_high && mu_mid < mu_target) ||
           (mu_low >  mu_high && mu_mid > mu_target)) {
            c_low = c_mid;
            mu_low = mu_mid;
        } else {
            c_high = c_mid;
            mu_high = mu_mid;
        }
    }
    return clamp_concentration(phase_id, 0.5 * (c_low + c_high));
}

/* ----------------------------------------------------------------
 * solve_local_equilibrium
 *
 * 输入：
 *   phi_vals[]     各相体积分数（会被 floor 修改，调用者需自行保存原始值）
 *   C_total_val    该网格点总浓度
 *
 * 输出：
 *   c_eq[]         各相局部平衡浓度
 *   mu_star        公共化学势
 *   phi_real_flag  [新增] 各相在 floor 前是否真实存在（1=真实，0=floor补）
 *                  数组长度需 >= NUM_COMP+1，下标 1..NUM_COMP
 * ---------------------------------------------------------------- */
void solve_local_equilibrium(double *phi_vals, double C_total_val,
                              double *c_eq, double *mu_star,
                              int *phi_real_flag)
{
    int n, t;
    double phi_sum_type[4] = {0.0, 0.0, 0.0, 0.0};
    double phi_real_type[4] = {0.0, 0.0, 0.0, 0.0};
    int rep_phase[4] = {-1, -1, -1, -1};
    int real_types[4];
    int n_real_types = 0;

    for(n = 1; n <= NUM_COMP; n++) {
        c_eq[n] = 0.0;
        phi_real_flag[n] = 0;
    }
    *mu_star = 0.0;

    for(n = 1; n <= NUM_COMP; n++) {
        if(!phase_active[n]) continue;
        t = get_thermo_type(n);
        phi_sum_type[t] += phi_vals[n];
        if(rep_phase[t] < 0) rep_phase[t] = n;
    }

    for(t = 0; t < 4; t++)
        phi_real_type[t] = (phi_sum_type[t] > 0.0) ? phi_sum_type[t] : 0.0;

    for(n = 1; n <= NUM_COMP; n++) {
        if(!phase_active[n]) continue;
        if(phi_vals[n] > 0.0) {
            phi_real_flag[n] = 1;
        }
    }

    for(t = 0; t < 4; t++) {
        if(phi_real_type[t] > 0.0)
            real_types[n_real_types++] = t;
    }

    if(n_real_types == 0) {
        for(n = 1; n <= NUM_COMP; n++)
            if(phase_active[n]) c_eq[n] = clamp_concentration(n, C_total_val);
        return;
    }

    if(n_real_types == 1) {
        int t0 = real_types[0];
        int p0 = rep_phase[t0];
        double c0 = clamp_concentration(p0, C_total_val);
        *mu_star = get_mu(p0, c0);
        for(n = 1; n <= NUM_COMP; n++) {
            if(!phase_active[n]) continue;
            if(get_thermo_type(n) == t0)
                c_eq[n] = clamp_concentration(n, c0);
        }
        return;
    }

    double wsum = 0.0;
    for(int a = 0; a < n_real_types; a++)
        wsum += phi_real_type[real_types[a]];
    if(wsum <= 0.0) wsum = 1.0;

    double mu_lo = 1e300, mu_hi = -1e300;
    for(int a = 0; a < n_real_types; a++) {
        t = real_types[a];
        int p = rep_phase[t];
        double c_low, c_high;
        get_concentration_bounds(p, &c_low, &c_high);
        double m_low = get_mu(p, c_low);
        double m_high = get_mu(p, c_high);
        if(m_low  < mu_lo) mu_lo = m_low;
        if(m_high < mu_lo) mu_lo = m_high;
        if(m_low  > mu_hi) mu_hi = m_low;
        if(m_high > mu_hi) mu_hi = m_high;
    }
    double margin = 0.05 * (mu_hi - mu_lo);
    if(margin < 1.0) margin = 1.0;
    mu_lo -= margin;
    mu_hi += margin;

    double mu_mid = 0.5 * (mu_lo + mu_hi);
    for(int iter = 0; iter < 160; iter++) {
        mu_mid = 0.5 * (mu_lo + mu_hi);
        double C_calc = 0.0;
        for(int a = 0; a < n_real_types; a++) {
            t = real_types[a];
            int p = rep_phase[t];
            double w = phi_real_type[t] / wsum;
            C_calc += w * invert_mu_to_c(p, mu_mid);
        }
        if(fabs(C_calc - C_total_val) < 1e-10 || fabs(mu_hi - mu_lo) < 1e-10)
            break;
        if(C_calc < C_total_val) mu_lo = mu_mid;
        else                     mu_hi = mu_mid;
    }
    *mu_star = mu_mid;

    double c_type[4] = {0.0, 0.0, 0.0, 0.0};
    int type_active[4] = {0, 0, 0, 0};
    for(int a = 0; a < n_real_types; a++) {
        t = real_types[a];
        c_type[t] = invert_mu_to_c(rep_phase[t], mu_mid);
        type_active[t] = 1;
    }
    for(n = 1; n <= NUM_COMP; n++) {
        if(!phase_active[n]) continue;
        t = get_thermo_type(n);
        if(type_active[t]) c_eq[n] = clamp_concentration(n, c_type[t]);
    }
}

 
 
/* ================================================================
 * extrapolate_ceq_zero_phi
 *
 * 对每个相，把 phi_real_flag==0 的格点替换为最近有效边界层的平均 CI。
 * 边界层由有效区与无效区相邻的点组成，因此可以跟随曲线界面。
 * ================================================================ */
 void extrapolate_ceq_zero_phi(int **phi_real_field, int blocksize, int mpi_rank, int mpi_size)
{
    int local_phys_size = blocksize * Nz;
    int global_phys_size = Nx * Nz;
    int *local_flag = (int*)malloc(local_phys_size * sizeof(int));
    int *global_flag = (int*)malloc(global_phys_size * sizeof(int));
    double *local_ci = (double*)malloc(local_phys_size * sizeof(double));
    double *global_ci = (double*)malloc(global_phys_size * sizeof(double));

    if(!local_flag || !global_flag || !local_ci || !global_ci) {
        free(local_flag); free(global_flag); free(local_ci); free(global_ci);
        return;
    }

    for(int n=1; n<=NUM_COMP; n++) {
        if(!phase_active[n]) continue;

        int idx = 0;
        for(int i=1; i<=blocksize; i++) {
            for(int k=1; k<=Nz; k++) {
                local_flag[idx] = phi_real_field[n][i*(Nz+2)+k] ? 1 : 0;
                local_ci[idx] = clamp_concentration(n, CI(n,i,k));
                idx++;
            }
        }

        MPI_Allgather(local_flag, local_phys_size, MPI_INT,
                      global_flag, local_phys_size, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(local_ci, local_phys_size, MPI_DOUBLE,
                      global_ci, local_phys_size, MPI_DOUBLE, MPI_COMM_WORLD);

        int global_min_x = Nx;
        int global_max_x = -1;
        for(int gx=0; gx<Nx; gx++) {
            for(int z=0; z<Nz; z++) {
                if(global_flag[gx*Nz + z]) {
                    if(gx < global_min_x) global_min_x = gx;
                    if(gx > global_max_x) global_max_x = gx;
                }
            }
        }
        if(global_max_x < global_min_x) continue;

        double mid_x = 0.5 * (global_min_x + global_max_x);
        double stats[4] = {0.0, 0.0, 0.0, 0.0};

        for(int gx=0; gx<Nx; gx++) {
            for(int z=0; z<Nz; z++) {
                int gidx = gx*Nz + z;
                if(!global_flag[gidx]) continue;
                int boundary = 0;
                if(gx == 0 || !global_flag[(gx-1)*Nz + z]) boundary = 1;
                if(gx == Nx-1 || !global_flag[(gx+1)*Nz + z]) boundary = 1;
                if(z == 0 || !global_flag[gx*Nz + (z-1)]) boundary = 1;
                if(z == Nz-1 || !global_flag[gx*Nz + (z+1)]) boundary = 1;
                if(!boundary) continue;

                double cval = clamp_concentration(n, global_ci[gidx]);
                if((double)gx <= mid_x) {
                    stats[0] += cval;
                    stats[2] += 1.0;
                } else {
                    stats[1] += cval;
                    stats[3] += 1.0;
                }
            }
        }

        double left_avg  = (stats[2] > 0.0) ? stats[0] / stats[2] : 0.0;
        double right_avg = (stats[3] > 0.0) ? stats[1] / stats[3] : left_avg;
        if(stats[2] <= 0.0) left_avg = right_avg;

        left_avg = clamp_concentration(n, left_avg);
        right_avg = clamp_concentration(n, right_avg);

        for(int i=1; i<=blocksize; i++) {
            int gx = mpi_rank * blocksize + (i - 1);
            for(int k=1; k<=Nz; k++) {
                if(phi_real_field[n][i*(Nz+2)+k]) continue;
                CI(n,i,k) = ((double)gx <= mid_x) ? left_avg : right_avg;
            }
        }
    }

    free(local_flag); free(global_flag); free(local_ci); free(global_ci);
}

/* ======================================================
 * time_step
 * 消亡相（phase_active[n]==0）完全跳过
 * ====================================================== */
int time_step(double **phi, double **ci, double **lap_phi, double **chem_pot,
              double **phi_t, double *c_total, double *c_t,
              double *graphic_x, double *graphic_z,int **phi_real_field,
              double *comm_buf, double *comm_buf1, double *f3, double h, double alpha,
              int mpi_rank, int mpi_size, unsigned long iloop)
{
    int i, k, n, j;
    int blocksize=Nx/mpi_size;
    double h2=h*h;

    int thermo_type[NUM_COMP+2];
    for(n=1;n<=NUM_COMP;n++) thermo_type[n]=get_thermo_type(n);
    // int *phi_real_field[NUM_COMP+2];
    // for(n=1; n<=NUM_COMP; n++)
    // phi_real_field[n] = (int*)calloc((blocksize+2)*(Nz+2), sizeof(int));

    
    for(i=1; i<=blocksize; i++) {
        for(k=1; k<=Nz; k++) {
            double phi_pre[NUM_COMP+2];
            for(n=1; n<=NUM_COMP; n++)
                phi_pre[n] = phase_active[n] ? PHI(n,i,k) : 0.0;
            double c_eq_pre[NUM_COMP+2];
            double mu_pre;
            int flag_pre[NUM_COMP+2];
            solve_local_equilibrium(phi_pre, C_total(i,k), c_eq_pre, &mu_pre, flag_pre);
            // 同时存储 flag
            for(n=1; n<=NUM_COMP; n++)
                phi_real_field[n][i*(Nz+2)+k] = flag_pre[n];
            //solve_local_equilibrium(phi_pre, C_total(i,k), c_eq_pre, &mu_pre);
            for(n=1; n<=NUM_COMP; n++) {
                if(!phase_active[n]) continue;
                CI(n,i,k) = c_eq_pre[n];   /* 覆盖为当前步平衡值 */
            }
        }
    }
    /* 预更新后先补Z向ghost，再同步X向CI/flag。这样外推在rank边界
     * 可以从相邻rank的真实相标志和CI值继续扫描。 */
    for(i=0; i<=blocksize+1; i++) {
        for(n=1; n<=NUM_COMP; n++) {
            if(!phase_active[n]) continue;
            CI(n,i,0)    = CI(n,i,1);
            CI(n,i,Nz+1) = CI(n,i,Nz);
        }
    }
    xdir_mpi_sr(phi, ci, c_total, graphic_x, graphic_z,
                phi_real_field, comm_buf, comm_buf1, mpi_rank, mpi_size);

    extrapolate_ceq_zero_phi(phi_real_field, blocksize, mpi_rank, mpi_size);
    filter_ci_z_after_extrapolation(blocksize);

    /* 外推会修改物理域CI；再次同步X向ghost，供后续边界通量使用。 */
    for(i=0; i<=blocksize+1; i++) {
        for(n=1; n<=NUM_COMP; n++) {
            if(!phase_active[n]) continue;
            CI(n,i,0)    = CI(n,i,1);
            CI(n,i,Nz+1) = CI(n,i,Nz);
        }
    }
    xdir_mpi_sr(phi, ci, c_total, graphic_x, graphic_z,
                phi_real_field, comm_buf, comm_buf1, mpi_rank, mpi_size);

    if(DEBUG_AB_KKS && iloop < DEBUG_EVERY_STEP_UNTIL)
        write_ab_kks_debug("pregrad", iloop+1, debug_folder1, blocksize, mpi_rank);
    

    /* [MOD-1-BC] 预更新后立即同步Z方向ghost层
    * X方向ghost层（i=0, i=blocksize+1）由main里[MOD-2]的
    * xdir_mpi_sr在time_step调用前已同步，此处只补Z方向 */
    for(i=0; i<=blocksize+1; i++) {
        for(n=1; n<=NUM_COMP; n++) {
            if(!phase_active[n]) continue;
            CI(n,i,0)    = CI(n,i,1);
            CI(n,i,Nz+1) = CI(n,i,Nz);
        }
    }

    /* ── 主循环：计算相场演化和浓度通量 ── */
    for(i=1;i<=blocksize;i++) {
        for(k=1;k<=Nz;k++) {
            double C_tot=C_total(i,k);

            double phi_local[NUM_COMP+2];
            double lap_local[NUM_COMP+2];
            double ci_local[NUM_COMP+2];

            for(n=1;n<=NUM_COMP;n++) {
                if(!phase_active[n]) { phi_local[n]=0.0; lap_local[n]=0.0; ci_local[n]=0.0; continue; }
                phi_local[n]=PHI(n,i,k);
                double edge_sum = PHI(n,i+1,k) + PHI(n,i-1,k)
                                + PHI(n,i,k+1) + PHI(n,i,k-1);
                double corner_sum = PHI(n,i+1,k+1) + PHI(n,i+1,k-1)
                                  + PHI(n,i-1,k+1) + PHI(n,i-1,k-1);
                lap_local[n]=(4*edge_sum + corner_sum- 20*phi_local[n])/(6*h2);
                LAP(n,i,k)=lap_local[n];
                ci_local[n]=CI(n,i,k);   /* 已是预更新后的值 */
            }

            /* [MOD-1] solve_local_equilibrium 仍保留，用于获取 mu_star
             * c_eq 与预更新结果相同，但 mu_star 是相场演化需要的 */
            double c_eq[NUM_COMP+2];
            double mu_star;
            int flag_dummy[NUM_COMP+2];
            solve_local_equilibrium(phi_local, C_tot, c_eq, &mu_star, flag_dummy);
                // flag_dummy 不使用，flag 已在预更新时记录
            //solve_local_equilibrium(phi_local,C_tot,c_eq,&mu_star);
            // [MOD-1-BC] z方向ghost层同步之后，立即外推
            //extrapolate_ceq_zero_phi(phi_real_field, blocksize);
            for(n=1;n<=NUM_COMP;n++) {
                if(!phase_active[n]) continue;
                ci_local[n]=c_eq[n];   /* 与预更新一致 */
                
                //ci_local[n] = invert_mu_to_c(n, mu_pre);  // 无该相：用当前mu外推，保持连续
            }

            double phi_al_x=0.5*(PHI(1,i+1,k) + PHI(1,i,k));
            double phi_mg_x=0.5*(PHI(NUM_COMP,i+1,k) + PHI(NUM_COMP,i,k));
            double phi_beta_x=0.0, phi_gamma_x=0.0;
            double phi_al_z=0.5*(PHI(1,i,k+1) + PHI(1,i,k));
            double phi_mg_z=0.5*(PHI(NUM_COMP,i,k+1) + PHI(NUM_COMP,i,k));
            double phi_beta_z=0.0, phi_gamma_z=0.0;
            for(n=2; n<=1+NUM_BETA; n++) {
                if(!phase_active[n]) continue;
                phi_beta_x += 0.5*(PHI(n,i+1,k) + PHI(n,i,k));
                phi_beta_z += 0.5*(PHI(n,i,k+1) + PHI(n,i,k));
            }
            for(n=2+NUM_BETA; n<=1+NUM_BETA+NUM_GAMMA; n++) {
                if(!phase_active[n]) continue;
                phi_gamma_x += 0.5*(PHI(n,i+1,k) + PHI(n,i,k));
                phi_gamma_z += 0.5*(PHI(n,i,k+1) + PHI(n,i,k));
            }

            double D_x = concentration_diffusivity(phi_al_x, phi_beta_x, phi_gamma_x, phi_mg_x);
            double D_z = concentration_diffusivity(phi_al_z, phi_beta_z, phi_gamma_z, phi_mg_z);
            double inter_cx=0.0, inter_cz=0.0;
            for(n=1; n<=NUM_COMP; n++) {
                if(!phase_active[n]) continue;
                /* x方向：i+½ 处 */
                double phi_half_x = 0.5*(PHI(n,i+1,k) + PHI(n,i,k));
                if(phi_half_x > 1e-10) {
                    double grad_ci_x = (CI(n,i+1,k) - CI(n,i,k)) / h;
                    inter_cx += D_x * phi_half_x * grad_ci_x;
                }

                /* z方向：有限差分面通量，不再使用Fourier周期导数 */
                double phi_half_z = 0.5*(PHI(n,i,k+1) + PHI(n,i,k));
                if(phi_half_z > 1e-10) {
                    double grad_ci_z = (CI(n,i,k+1) - CI(n,i,k)) / h;
                    inter_cz += D_z * phi_half_z * grad_ci_z;
                }
            }

            double phi_type_sum[4] = {0.0, 0.0, 0.0, 0.0};
            double lap_type_sum[4] = {0.0, 0.0, 0.0, 0.0};
            double c_type_val[4] = {0.0, 0.0, 0.0, 0.0};
            double c_type_weight[4] = {0.0, 0.0, 0.0, 0.0};
            double f_type_val[4] = {0.0, 0.0, 0.0, 0.0};
            int type_has_phase[4] = {0, 0, 0, 0};

            for(n=1;n<=NUM_COMP;n++) {
                if(!phase_active[n]) continue;
                int t_n=thermo_type[n];
                phi_type_sum[t_n] += phi_local[n];
                lap_type_sum[t_n] += lap_local[n];
                type_has_phase[t_n] = 1;
                if(phi_local[n] > 0.0) {
                    c_type_val[t_n] += phi_local[n] * c_eq[n];
                    c_type_weight[t_n] += phi_local[n];
                }
            }
            for(int tt=0; tt<4; tt++) {
                if(c_type_weight[tt] > 0.0) c_type_val[tt] /= c_type_weight[tt];
            }
            if(type_has_phase[IDX_AL])
                f_type_val[IDX_AL] = f_Al(c_type_val[IDX_AL]);
            if(type_has_phase[IDX_BETA])
                f_type_val[IDX_BETA] = f_Beta(c_type_val[IDX_BETA]);
            if(type_has_phase[IDX_GAMMA])
                f_type_val[IDX_GAMMA] = f_Gamma(c_type_val[IDX_GAMMA]);
            if(type_has_phase[IDX_MG])
                f_type_val[IDX_MG] = f_Mg(c_type_val[IDX_MG]);

            int present[NUM_COMP+2];
            int n_present = 0;
            for(n=1; n<=NUM_COMP; n++) {
                present[n] = 0;
                if(!phase_active[n]) continue;
                if(phi_local[n] > 1e-10 || fabs(lap_local[n]) > 1e-10) {
                    present[n] = 1;
                    n_present++;
                }
            }

            for(n=1;n<=NUM_COMP;n++) {
                if(!present[n]) { PHI_t(n,i,k)=0.0; continue; }
                double p_k=phi_local[n];

                double lap_k=lap_local[n];
                double c_n=ci_local[n];
                double f_n;
                int t_n=thermo_type[n];
                if(t_n==IDX_AL) f_n=f_Al(c_n);
                else if(t_n==IDX_MG) f_n=f_Mg(c_n);
                else if(t_n==IDX_BETA) f_n=f_Beta(c_n);
                else f_n=f_Gamma(c_n);

                double interaction_sum=0.0;

                for(j=1;j<=NUM_COMP;j++) {
                    if(j==n || !present[j]) continue;
                    int t_j=thermo_type[j];

                    double p_j=phi_local[j];

                    double lap_j=lap_local[j];
                    double c_j=ci_local[j];
                    double f_j;
                    if(t_j==IDX_AL) f_j=f_Al(c_j);
                    else if(t_j==IDX_MG) f_j=f_Mg(c_j);
                    else if(t_j==IDX_BETA) f_j=f_Beta(c_j);
                    else f_j=f_Gamma(c_j);

                    double M_kj=1.0, W_kj=1.0, a2_kj=2.0;
                    int beta_gamma=((t_n==IDX_BETA&&t_j==IDX_GAMMA)||(t_n==IDX_GAMMA&&t_j==IDX_BETA));
                    int al_beta=((t_n==IDX_AL&&t_j==IDX_BETA)||(t_n==IDX_BETA&&t_j==IDX_AL));
                    int gamma_mg=((t_n==IDX_GAMMA&&t_j==IDX_MG)||(t_n==IDX_MG&&t_j==IDX_GAMMA));
                    int same_imc=(t_n==t_j && t_n!=IDX_AL && t_n!=IDX_MG);
                    int any_imc=(t_n!=IDX_AL&&t_n!=IDX_MG)||(t_j!=IDX_AL&&t_j!=IDX_MG);

                    if(beta_gamma)       { M_kj=M_BETA_GAMMA; W_kj=1.0; a2_kj=2.0; }
                    else if(al_beta)     { M_kj=M_AL_BETA;    W_kj=1.0; a2_kj=2.0; }
                    else if(gamma_mg)    { M_kj=M_GAMMA_MG;   W_kj=1.0; a2_kj=2.0; }
                    else if(same_imc)    { M_kj=M_SAME_IMC;   W_kj=1.0; a2_kj=2.0; }
                    else if(any_imc)     { M_kj=M_OTHER_IMC;  W_kj=1.0; a2_kj=2.0; }

                    double phase_term = W_kj * (p_k - p_j);
                    double grad_term = 0.5 * a2_kj * (lap_k - lap_j);
                    double dg=(f_n-c_n*mu_star)-(f_j-c_j*mu_star);
                    interaction_sum += 2.0 * M_kj * (phase_term + grad_term)
                                     - CHEM_DRIVE_SCALE * sqrt(p_k * p_j) * (8.0/PI) * dg;
                }


                PHI_t(n,i,k)=(n_present > 0) ? interaction_sum / (double)n_present : 0.0;
            }

            GRAPHICX(i,k)=inter_cx;
            GRAPHICZ(i,k)=inter_cz;
        }
    }

    for(i=1; i<=blocksize; i++) {
        GRAPHICZ(i,0) = 0.0;
        GRAPHICZ(i,Nz) = 0.0;
    }

    for(i=1;i<=blocksize;i++) {
        for(k=1;k<=Nz;k++) {
            C_t(i,k)=(GRAPHICX(i,k)-GRAPHICX(i-1,k))/(h)
                    +(GRAPHICZ(i,k)-GRAPHICZ(i,k-1))/(h);
        }
    }

    for(i=1;i<=blocksize;i++) {
        for(k=1;k<=Nz;k++) {
            C_total(i,k)+=DT*C_MOBILITY*C_t(i,k);
            double phi_sum=0.0;
            for(n=1;n<=NUM_COMP;n++) {
                if(!phase_active[n]) {
                    PHI(n,i,k)=0.0;
                    continue;
                }
                double dphi=10*DT*PHI_t(n,i,k);
                if(dphi> 0.02) dphi= 0.02;
                if(dphi<-0.02) dphi=-0.02;
                double phi_new=PHI(n,i,k)+dphi;
                if(phi_new<0.0) phi_new=0.0;
                if(phi_new<PHI_ZERO_CUTOFF) phi_new=0.0;
                PHI(n,i,k)=phi_new;
                phi_sum+=phi_new;
            }
            if(phi_sum>1e-12) {
                double inv=1.0/phi_sum;
                for(n=1;n<=NUM_COMP;n++)
                    if(phase_active[n]) PHI(n,i,k)*=inv;
            } else {
                PHI(1,i,k)=1.0;
                for(n=2;n<=NUM_COMP;n++) PHI(n,i,k)=0.0;
            }
        }
    }
    refresh_local_ci_from_current_phi(phi_real_field, comm_buf, comm_buf1, mpi_rank, mpi_size);

    if(DEBUG_AB_KKS && iloop < DEBUG_EVERY_STEP_UNTIL)
        write_ab_kks_debug("postupdate", iloop+1, debug_folder1, blocksize, mpi_rank);

    for(i=1; i<=blocksize; i++) {
        for(k=1; k<=Nz; k++) {
            int didx = (i-1)*Nz + (k-1);

            diag_ci_al[didx]  = CI(1,i,k);
            diag_ci_beta[didx]= family_local_ci(IDX_BETA, i, k);
            diag_ci_gamma[didx]= family_local_ci(IDX_GAMMA, i, k);
            diag_ci_mg[didx]  = CI(NUM_COMP,i,k);
        }
    }

    return 0;
}

/* ======================================================
 * xdir_mpi_sr
 * ====================================================== */
int xdir_mpi_sr(double **phi, double **ci, double *c_total,
                double *graphic_x, double *graphic_z,
                int **phi_real_field, double *comm_buf, double *comm_buf1,
                int mpi_rank, int mpi_size)
{
    int blocksize=Nx/mpi_size;
    int buf_len=Nz+2;
    int offset_c  = NUM_COMP*buf_len;
    int offset_ci = offset_c  + buf_len;
    int offset_gx = offset_ci + NUM_COMP*buf_len;   /* graphic_x */
    int offset_gz = offset_gx + buf_len;             /* graphic_z */
    int total_size= offset_gz + buf_len;             /* 总长度 */

    int left =(mpi_rank==0)         ?MPI_PROC_NULL:mpi_rank-1;
    int right=(mpi_rank==mpi_size-1)?MPI_PROC_NULL:mpi_rank+1;

    if((!comm_buf || !comm_buf1) && mpi_size > 1) return -1;

    if(left!=MPI_PROC_NULL) {
        double *sbuf=comm_buf, *rbuf=comm_buf+total_size;
        for(int n=1;n<=NUM_COMP;n++) memcpy(sbuf+(n-1)*buf_len,&PHI(n,1,0),buf_len*sizeof(double));
        memcpy(sbuf+offset_c, &C_total(1,0),  buf_len*sizeof(double));
        for(int n=1;n<=NUM_COMP;n++) memcpy(sbuf+offset_ci+(n-1)*buf_len,&CI(n,1,0),buf_len*sizeof(double));
        memcpy(sbuf+offset_gx,&GRAPHICX(1,0), buf_len*sizeof(double));
        memcpy(sbuf+offset_gz,&GRAPHICZ(1,0), buf_len*sizeof(double));
        MPI_Sendrecv(sbuf,total_size,MPI_DOUBLE,left,0,
                     rbuf,total_size,MPI_DOUBLE,left,0,
                     MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        for(int n=1;n<=NUM_COMP;n++) memcpy(&PHI(n,0,0),rbuf+(n-1)*buf_len,buf_len*sizeof(double));
        memcpy(&C_total(0,0), rbuf+offset_c,  buf_len*sizeof(double));
        for(int n=1;n<=NUM_COMP;n++) memcpy(&CI(n,0,0),rbuf+offset_ci+(n-1)*buf_len,buf_len*sizeof(double));
        memcpy(&GRAPHICX(0,0),rbuf+offset_gx, buf_len*sizeof(double));
        memcpy(&GRAPHICZ(0,0),rbuf+offset_gz, buf_len*sizeof(double));
        if(phi_real_field) {
            for(int n=1; n<=NUM_COMP; n++) {
                MPI_Sendrecv(&phi_real_field[n][1*(Nz+2)], buf_len, MPI_INT, left, 1,
                             &phi_real_field[n][0*(Nz+2)], buf_len, MPI_INT, left, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    if(right!=MPI_PROC_NULL) {
        double *sbuf=comm_buf1, *rbuf=comm_buf1+total_size;
        for(int n=1;n<=NUM_COMP;n++) memcpy(sbuf+(n-1)*buf_len,&PHI(n,blocksize,0),buf_len*sizeof(double));
        memcpy(sbuf+offset_c, &C_total(blocksize,0),  buf_len*sizeof(double));
        for(int n=1;n<=NUM_COMP;n++) memcpy(sbuf+offset_ci+(n-1)*buf_len,&CI(n,blocksize,0),buf_len*sizeof(double));
        memcpy(sbuf+offset_gx,&GRAPHICX(blocksize,0), buf_len*sizeof(double));
        memcpy(sbuf+offset_gz,&GRAPHICZ(blocksize,0), buf_len*sizeof(double));
        MPI_Sendrecv(sbuf,total_size,MPI_DOUBLE,right,0,
                     rbuf,total_size,MPI_DOUBLE,right,0,
                     MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        for(int n=1;n<=NUM_COMP;n++) memcpy(&PHI(n,blocksize+1,0),rbuf+(n-1)*buf_len,buf_len*sizeof(double));
        memcpy(&C_total(blocksize+1,0), rbuf+offset_c,  buf_len*sizeof(double));
        for(int n=1;n<=NUM_COMP;n++) memcpy(&CI(n,blocksize+1,0),rbuf+offset_ci+(n-1)*buf_len,buf_len*sizeof(double));
        memcpy(&GRAPHICX(blocksize+1,0),rbuf+offset_gx, buf_len*sizeof(double));
        memcpy(&GRAPHICZ(blocksize+1,0),rbuf+offset_gz, buf_len*sizeof(double));
        if(phi_real_field) {
            for(int n=1; n<=NUM_COMP; n++) {
                MPI_Sendrecv(&phi_real_field[n][blocksize*(Nz+2)], buf_len, MPI_INT, right, 1,
                             &phi_real_field[n][(blocksize+1)*(Nz+2)], buf_len, MPI_INT, right, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    return 0;
}

/* ======================================================
 * set_bc
 * ====================================================== */
int set_bc(double **phi, double *c_total, double **ci,
           double h, int mpi_rank, int mpi_size)
{
    int i,k,n;
    int blocksize=Nx/mpi_size;
    for(i=0;i<=blocksize+1;i++) {
        for(n=1;n<=NUM_COMP;n++) {
            PHI(n,i,0)=PHI(n,i,2);       PHI(n,i,Nz+1)=PHI(n,i,Nz-1);
            CI(n,i,0)=CI(n,i,1);         CI(n,i,Nz+1)=CI(n,i,Nz);
        }
        C_total(i,0)=C_total(i,1);       C_total(i,Nz+1)=C_total(i,Nz);
    }
    if(mpi_rank==0)
        for(k=0;k<=Nz+1;k++) {
            for(n=1;n<=NUM_COMP;n++) { PHI(n,0,k)=PHI(n,2,k); CI(n,0,k)=CI(n,2,k); }
            C_total(0,k)=C_total(2,k);
        }
    if(mpi_rank==mpi_size-1)
        for(k=0;k<=Nz+1;k++) {
            for(n=1;n<=NUM_COMP;n++) { PHI(n,blocksize+1,k)=PHI(n,blocksize-1,k); CI(n,blocksize+1,k)=CI(n,blocksize-1,k); }
            C_total(blocksize+1,k)=C_total(blocksize-1,k);
        }
    return 0;
}

double family_local_ci(int type, int x, int z)
{
    int first, last;
    double fallback;
    if(type == IDX_BETA) {
        first = 2;
        last = 1 + NUM_BETA;
        fallback = C_BETA;
    } else if(type == IDX_GAMMA) {
        first = 2 + NUM_BETA;
        last = 1 + NUM_BETA + NUM_GAMMA;
        fallback = C_GAMMA;
    } else {
        return 0.0;
    }

    double sum_phi = 0.0;
    double sum_ci = 0.0;
    for(int n = first; n <= last; n++) {
        if(!phase_active[n]) continue;
        double p = PHI(n,x,z);
        if(p > 0.0) {
            sum_phi += p;
            sum_ci += p * CI(n,x,z);
        }
    }
    return (sum_phi > 0.0) ? (sum_ci / sum_phi) : fallback;
}

void write_ab_kks_debug(const char *stage, unsigned long step, char *folder1,
                        int blocksize, int mpi_rank)
{
    int target_x_min = 76, target_x_max = 82;
    int target_z_min = 156, target_z_max = 184;
    char filename[256];
    sprintf(filename, "%s/debug_ab_kks_%s_rank%d_step%lu.csv",
            folder1, stage, mpi_rank, step);
    FILE *fp = fopen(filename, "w");
    if(!fp) return;

    fprintf(fp, "gx,z,C_total,phi_al,phi_beta,phi_gamma,phi_mg,w_al,w_beta,w_gamma,w_mg,types,mu,cKKS_al,cKKS_beta,cKKS_gamma,cKKS_mg,CI_al,CI_beta,CI_gamma,CI_mg\n");

    for(int i=1; i<=blocksize; i++) {
        int gx = mpi_rank * blocksize + (i - 1);
        if(gx < target_x_min || gx > target_x_max) continue;
        for(int z0=target_z_min; z0<=target_z_max; z0++) {
            int k = z0 + 1;
            double phi_vals[NUM_COMP+2];
            double c_eq[NUM_COMP+2];
            int flags[NUM_COMP+2];
            double mu;
            double phi_type[4] = {0.0,0.0,0.0,0.0};
            double w_type[4] = {0.0,0.0,0.0,0.0};
            int type_flag[4] = {0,0,0,0};

            for(int n=1; n<=NUM_COMP; n++) {
                phi_vals[n] = phase_active[n] ? PHI(n,i,k) : 0.0;
                int tt = get_thermo_type(n);
                phi_type[tt] += phi_vals[n];
            }
            for(int tt=0; tt<4; tt++)
                w_type[tt] = (phi_type[tt] > 0.0) ? phi_type[tt] : 0.0;

            solve_local_equilibrium(phi_vals, C_total(i,k), c_eq, &mu, flags);
            for(int n=1; n<=NUM_COMP; n++)
                if(flags[n]) type_flag[get_thermo_type(n)] = 1;

            double c_kks_type[4] = {0.0,0.0,0.0,0.0};
            double wsum_type[4] = {0.0,0.0,0.0,0.0};
            double ci_type[4] = {CI(1,i,k),0.0,0.0,CI(NUM_COMP,i,k)};
            double ci_wsum[4] = {PHI(1,i,k),0.0,0.0,PHI(NUM_COMP,i,k)};
            for(int n=2; n<=1+NUM_BETA; n++) {
                if(!phase_active[n]) continue;
                double p = PHI(n,i,k);
                c_kks_type[IDX_BETA] += p * c_eq[n];
                wsum_type[IDX_BETA] += p * (flags[n] ? 1.0 : 0.0);
                ci_type[IDX_BETA] += p * CI(n,i,k);
                ci_wsum[IDX_BETA] += p;
            }
            for(int n=2+NUM_BETA; n<=1+NUM_BETA+NUM_GAMMA; n++) {
                if(!phase_active[n]) continue;
                double p = PHI(n,i,k);
                c_kks_type[IDX_GAMMA] += p * c_eq[n];
                wsum_type[IDX_GAMMA] += p * (flags[n] ? 1.0 : 0.0);
                ci_type[IDX_GAMMA] += p * CI(n,i,k);
                ci_wsum[IDX_GAMMA] += p;
            }
            c_kks_type[IDX_AL] = c_eq[1];
            c_kks_type[IDX_MG] = c_eq[NUM_COMP];
            if(wsum_type[IDX_BETA] > 0.0) c_kks_type[IDX_BETA] /= wsum_type[IDX_BETA];
            if(wsum_type[IDX_GAMMA] > 0.0) c_kks_type[IDX_GAMMA] /= wsum_type[IDX_GAMMA];
            if(ci_wsum[IDX_BETA] > 0.0) ci_type[IDX_BETA] /= ci_wsum[IDX_BETA];
            if(ci_wsum[IDX_GAMMA] > 0.0) ci_type[IDX_GAMMA] /= ci_wsum[IDX_GAMMA];

            fprintf(fp, "%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%d%d%d%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                    gx, z0, C_total(i,k),
                    phi_type[IDX_AL], phi_type[IDX_BETA], phi_type[IDX_GAMMA], phi_type[IDX_MG],
                    w_type[IDX_AL], w_type[IDX_BETA], w_type[IDX_GAMMA], w_type[IDX_MG],
                    type_flag[IDX_AL], type_flag[IDX_BETA], type_flag[IDX_GAMMA], type_flag[IDX_MG],
                    mu,
                    c_kks_type[IDX_AL], c_kks_type[IDX_BETA], c_kks_type[IDX_GAMMA], c_kks_type[IDX_MG],
                    ci_type[IDX_AL], ci_type[IDX_BETA], ci_type[IDX_GAMMA], ci_type[IDX_MG]);
        }
    }

    fclose(fp);
}

int epsout(double **phi, double *c_total, double **ci,double *c_t,
           double h, int mpi_rank, int mpi_size,
           int eps_tag, char *folder1)
{
    int blocksize        = Nx / mpi_size;
    int local_phys_size  = blocksize * Nz;
    int global_phys_size = Nx * Nz;
    char filename[256], dir_cmd[256];
    FILE *fp; int l, n;

    if(mpi_rank==0) {
        sprintf(dir_cmd, "mkdir -p %s", folder1);
        system(dir_cmd);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double *local_buf  = (double*)malloc(local_phys_size * sizeof(double));
    double *global_buf = (mpi_rank==0)
                        ?(double*)malloc(global_phys_size * sizeof(double))
                        :NULL;
    if(!local_buf) return -1;

    if(DEBUG_AB_KKS)
        write_ab_kks_debug("eps", (unsigned long)eps_tag, folder1, blocksize, mpi_rank);

    /* ---- 每个相的 phi VTK 输出已注释（改用 energy_epsout 输出总自由能场）----*/
    for (n=1;n<=NUM_COMP;n++) {
        int idx=0;
        for (int x=1;x<=blocksize;x++) for (int z=1;z<=Nz;z++) local_buf[idx++]=PHI(n,x,z);
        if (mpi_rank==0) {
            memcpy(global_buf,local_buf,local_phys_size*sizeof(double));
            for (l=1;l<mpi_size;l++) {
                MPI_Recv(&global_buf[l*local_phys_size],local_phys_size,MPI_DOUBLE,l,n,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
            sprintf(filename,"%s/phi_%d_step%d.vtk",folder1,n,eps_tag);
            fp=fopen(filename,"w");
            if (fp) {
                fprintf(fp,"# vtk DataFile Version 3.0\nPhase %d Step %d\nASCII\n",n,eps_tag);
                fprintf(fp,"DATASET STRUCTURED_POINTS\nDIMENSIONS %d %d 1\n",Nx,Nz);
                fprintf(fp,"ORIGIN 0 0 0\nSPACING %f %f 1.0\n",h,h);
                fprintf(fp,"POINT_DATA %d\nSCALARS phi_%d float 1\nLOOKUP_TABLE default\n",global_phys_size,n);
                for (int zo=0;zo<Nz;zo++) for (int xo=0;xo<Nx;xo++) fprintf(fp,"%f\n",global_buf[xo*Nz+zo]);
                fclose(fp);
            }
        } else {
            MPI_Send(local_buf,local_phys_size,MPI_DOUBLE,0,n,MPI_COMM_WORLD);
        }
    }
    /*---- 注释结束 ---- */

    /* ================================================================
     * 文件1：四类热力学相 ID 场
     *   0 = Al   (相1)
     *   1 = Beta (相2..NUM_BETA+1)
     *   2 = Gamma(相NUM_BETA+2..NUM_BETA+NUM_GAMMA+1)
     *   3 = Mg   (相NUM_COMP)
     * ================================================================ */
    {
        int idx=0;
        for(int x=1; x<=blocksize; x++) {
            for(int z=1; z<=Nz; z++) {
                /* 找主导相 */
                int dom=1; double vmax=0.0;
                for(n=1; n<=NUM_COMP; n++)
                    if(PHI(n,x,z) > vmax) { vmax=PHI(n,x,z); dom=n; }

                /* 映射到4类 */
                int cls;
                if(dom == 1)                              cls = 0; /* Al   */
                else if(dom <= 1 + NUM_BETA)              cls = 1; /* Beta */
                else if(dom <= 1 + NUM_BETA + NUM_GAMMA)  cls = 2; /* Gamma*/
                else                                      cls = 3; /* Mg   */

                local_buf[idx++] = (double)cls;
            }
        }

        int p_tag = 100;
        if(mpi_rank==0) {
            memcpy(global_buf, local_buf, local_phys_size*sizeof(double));
            for(l=1; l<mpi_size; l++)
                MPI_Recv(&global_buf[l*local_phys_size], local_phys_size,
                         MPI_DOUBLE, l, p_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            sprintf(filename, "%s/phase_class_step%d.vtk", folder1, eps_tag);
            fp = fopen(filename, "w");
            if(fp) {
                fprintf(fp, "# vtk DataFile Version 3.0\n");
                fprintf(fp, "Phase Class Step %d\nASCII\n", eps_tag);
                fprintf(fp, "DATASET STRUCTURED_POINTS\n");
                /* 修正：DIMENSIONS = Nx(x方向) Nz(z方向) 1 */
                fprintf(fp, "DIMENSIONS %d %d 1\n", Nx, Nz);
                fprintf(fp, "ORIGIN 0 0 0\nSPACING %f %f 1.0\n", h, h);
                fprintf(fp, "POINT_DATA %d\n", global_phys_size);
                fprintf(fp, "SCALARS phase_class float 1\nLOOKUP_TABLE default\n");
                /*
                 * 修正：VTK StructuredPoints 数据排列为 x 变化最快（行优先）
                 * 即: for z from 0 to Nz-1:
                 *       for x from 0 to Nx-1:
                 *         write data[x][z]
                 * 注意 local_buf 的存储是 [x][z]，即 global_buf[x*Nz+z]
                 */
                for(int zo=0; zo<Nz; zo++)
                    for(int xo=0; xo<Nx; xo++)
                        fprintf(fp, "%.0f\n", global_buf[xo*Nz+zo]);
                fclose(fp);
            }
        } else {
            MPI_Send(local_buf, local_phys_size, MPI_DOUBLE, 0, p_tag, MPI_COMM_WORLD);
        }
    }

    /* ================================================================
     * 文件2：界面强度场
     *   iface(x,z) = 1 - max_n(phi_n)
     *   纯相内部 max_phi→1  → iface→0（透明）
     *   界面处   max_phi→0.5 → iface→0.5（半透明白色）
     *   ParaView 里用 Opacity 映射渲染白色轮廓线
     * ================================================================ */
    {
        int idx=0;
        for(int x=1; x<=blocksize; x++) {
            for(int z=1; z<=Nz; z++) {
                double vmax=0.0;
                for(n=1; n<=NUM_COMP; n++)
                    if(PHI(n,x,z) > vmax) vmax = PHI(n,x,z);
                /* 界面强度：内部=0，界面=最大值约0.5时强度最大 */
                local_buf[idx++] = 1.0 - vmax;
            }
        }

        int i_tag = 101;
        if(mpi_rank==0) {
            memcpy(global_buf, local_buf, local_phys_size*sizeof(double));
            for(l=1; l<mpi_size; l++)
                MPI_Recv(&global_buf[l*local_phys_size], local_phys_size,
                         MPI_DOUBLE, l, i_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            sprintf(filename, "%s/interface_step%d.vtk", folder1, eps_tag);
            fp = fopen(filename, "w");
            if(fp) {
                fprintf(fp, "# vtk DataFile Version 3.0\n");
                fprintf(fp, "Interface Intensity Step %d\nASCII\n", eps_tag);
                fprintf(fp, "DATASET STRUCTURED_POINTS\n");
                fprintf(fp, "DIMENSIONS %d %d 1\n", Nx, Nz);
                fprintf(fp, "ORIGIN 0 0 0\nSPACING %f %f 1.0\n", h, h);
                fprintf(fp, "POINT_DATA %d\n", global_phys_size);
                fprintf(fp, "SCALARS interface float 1\nLOOKUP_TABLE default\n");
                for(int zo=0; zo<Nz; zo++)
                    for(int xo=0; xo<Nx; xo++)
                        fprintf(fp, "%g\n", global_buf[xo*Nz+zo]);
                fclose(fp);
            }
        } else {
            MPI_Send(local_buf, local_phys_size, MPI_DOUBLE, 0, i_tag, MPI_COMM_WORLD);
        }
    }

    /* ================================================================
     * 文件3：c_total 浓度场（保留原有输出）
     * ================================================================ */
    {
        int idx=0;
        for(int x=1; x<=blocksize; x++)
            for(int z=1; z<=Nz; z++)
                local_buf[idx++] = C_total(x,z);

        int c_tag = 102;
        if(mpi_rank==0) {
            memcpy(global_buf, local_buf, local_phys_size*sizeof(double));
            for(l=1; l<mpi_size; l++)
                MPI_Recv(&global_buf[l*local_phys_size], local_phys_size,
                         MPI_DOUBLE, l, c_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            sprintf(filename, "%s/con_step%d.vtk", folder1, eps_tag);
            fp = fopen(filename, "w");
            if(fp) {
                fprintf(fp, "# vtk DataFile Version 3.0\n");
                fprintf(fp, "Concentration Step %d\nASCII\n", eps_tag);
                fprintf(fp, "DATASET STRUCTURED_POINTS\n");
                fprintf(fp, "DIMENSIONS %d %d 1\n", Nx, Nz);
                fprintf(fp, "ORIGIN 0 0 0\nSPACING %f %f 1.0\n", h, h);
                fprintf(fp, "POINT_DATA %d\n", global_phys_size);
                fprintf(fp, "SCALARS concentration float 1\nLOOKUP_TABLE default\n");
                for(int zo=0; zo<Nz; zo++)
                    for(int xo=0; xo<Nx; xo++)
                        fprintf(fp, "%g\n", global_buf[xo*Nz+zo]);
                fclose(fp);
            }
        } else {
            MPI_Send(local_buf, local_phys_size, MPI_DOUBLE, 0, c_tag, MPI_COMM_WORLD);
        }
    }

   
/* ================================================================
 * 文件17~20：time_step内预更新后的四类局部浓度场（计算过程真实值）
 *   来源：time_step内[MOD-1-BC]之后立即记录，是通量计算时用到的CI
 * ================================================================ */
{
    const char *diag_ci_names[4] = {
        "diag_ci_Al", "diag_ci_Beta", "diag_ci_Gamma", "diag_ci_Mg"
    };
    const int diag_ci_tags[4] = {300, 301, 302, 303};
    int diag_ci_types[4] = {IDX_AL, IDX_BETA, IDX_GAMMA, IDX_MG};

    int dd;
    for(dd=0; dd<4; dd++) {
        fill_type_ci_buffer(diag_ci_types[dd], local_buf, blocksize, mpi_rank, 0);

        if(mpi_rank==0) {
            memcpy(global_buf, local_buf,
                   local_phys_size*sizeof(double));
            for(l=1; l<mpi_size; l++)
                MPI_Recv(&global_buf[l*local_phys_size],
                         local_phys_size, MPI_DOUBLE, l,
                         diag_ci_tags[dd], MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

            sprintf(filename, "%s/%s_step%d.vtk",
                    folder1, diag_ci_names[dd], eps_tag);
            fp = fopen(filename, "w");
            if(fp) {
                fprintf(fp, "# vtk DataFile Version 3.0\n");
                fprintf(fp, "%s Step %d\nASCII\n",
                        diag_ci_names[dd], eps_tag);
                fprintf(fp, "DATASET STRUCTURED_POINTS\n");
                fprintf(fp, "DIMENSIONS %d %d 1\n", Nx, Nz);
                fprintf(fp, "ORIGIN 0 0 0\n");
                fprintf(fp, "SPACING %f %f 1.0\n", h, h);
                fprintf(fp, "POINT_DATA %d\n", global_phys_size);
                fprintf(fp, "SCALARS %s float 1\n",
                        diag_ci_names[dd]);
                fprintf(fp, "LOOKUP_TABLE default\n");
                for(int zo=0; zo<Nz; zo++)
                    for(int xo=0; xo<Nx; xo++)
                        fprintf(fp, "%f\n",
                                global_buf[xo*Nz+zo]);
                fclose(fp);
            }
        } else {
            MPI_Send(local_buf, local_phys_size, MPI_DOUBLE,
                     0, diag_ci_tags[dd], MPI_COMM_WORLD);
        }
    }
}

    free(local_buf);
    if(global_buf) free(global_buf);
    if(mpi_rank==0) printf("VTK output step %d done.\n", eps_tag);
    return 0;
}

/* ======================================================
 * energy_epsout
 * ====================================================== */

int energy_epsout(double **phi, double **ci, double h,
                  int mpi_rank, int mpi_size, int eps_tag, char *folder1)
{
    int i,k,n,j;
    int blocksize        = Nx/mpi_size;
    int local_phys_size  = blocksize*Nz;
    int global_phys_size = Nx*Nz;
    double h2=h*h;
    char filename[256]; FILE *fp; int l;

    int thermo_type[NUM_COMP+2];
    for(n=1;n<=NUM_COMP;n++) thermo_type[n]=get_thermo_type(n);

    double *local_energy  = (double*)calloc(local_phys_size, sizeof(double));
    double *global_energy = (mpi_rank==0)
                           ?(double*)malloc(global_phys_size*sizeof(double))
                           :NULL;
    if(!local_energy) return -1;

    for(i=1; i<=blocksize; i++) {
        for(k=1; k<=Nz; k++) {

            /* 预计算拉普拉斯（所有活跃相）*/
            double lap[NUM_COMP+2];
            for(n=1; n<=NUM_COMP; n++) {
                if(!phase_active[n]) { lap[n]=0.0; continue; }
                double edge_sum = PHI(n,i+1,k)+PHI(n,i-1,k)
                                + PHI(n,i,k+1)+PHI(n,i,k-1);
                double corner_sum = PHI(n,i+1,k+1)+PHI(n,i+1,k-1)
                                  + PHI(n,i-1,k+1)+PHI(n,i-1,k-1);
                lap[n] = (4.0*edge_sum + corner_sum - 20.0*PHI(n,i,k)) / (6.0*h2);
            }

            double f_total = 0.0;

            for(n=1; n<=NUM_COMP; n++) {
                if(!phase_active[n]) continue;
                double p_n = PHI(n,i,k);
                if(p_n < 1e-10) continue;

                for(j=n+1; j<=NUM_COMP; j++) {
                    if(!phase_active[j]) continue;
                    double p_j = PHI(j,i,k);
                    if(p_j < 1e-10) continue;

                    /* 交互参数（与 time_step 一致）*/
                    double W_nj=1.0, a2_nj=1.0;
                    int t_n=thermo_type[n], t_j=thermo_type[j];
                    int same_imc  = (t_n==t_j&&t_n!=IDX_AL&&t_n!=IDX_MG);
                    int cross_imc = ((t_n==IDX_BETA&&t_j==IDX_GAMMA)
                                   ||(t_n==IDX_GAMMA&&t_j==IDX_BETA));
                    int any_imc   = (t_n!=IDX_AL&&t_n!=IDX_MG)
                                   ||(t_j!=IDX_AL&&t_j!=IDX_MG);
                    if(same_imc)       { W_nj=4.0; a2_nj=2.0; }
                    else if(cross_imc) { W_nj=2.0; a2_nj=1.5; }
                    else if(any_imc)   { W_nj=1.2; a2_nj=1.0; }

                    /* 第一项：W·(φₙ²-2φₙ³+φₙ⁴)  只算一次（不需要求和）*/
                    double p2=p_n, p3=p2*p_n, p4=p3*p_n;
                    f_total += 0.0*W_nj * (2*p2 - 6.0*p3 + 4*p4);

                    /* 第二项：Σₙ<ⱼ 2W·φₙ²·φⱼ² */
                    f_total +=0.0* 2.0 * W_nj * p_n * p_j*p_j;

                    /* 第三项：-a²·∇²φₙ */
                    f_total += -a2_nj * lap[n];
                }
            }

            local_energy[(i-1)*Nz+(k-1)] = f_total;
        }
    }

    /* ---- 打印全场总自由能 ---- */
    double local_sum=0.0;
    for(int idx=0; idx<local_phys_size; idx++) local_sum += local_energy[idx];
    local_sum *= h2;
    double global_sum=0.0;
    MPI_Reduce(&local_sum,&global_sum,1,MPI_DOUBLE,MPI_SUM,MPI_MASTER,MPI_COMM_WORLD);
    if(mpi_rank==0)
        printf("[Energy] step=%d  Total=%.6e\n", eps_tag, global_sum);

    /* ---- MPI 汇聚并写 VTK ---- */
    int e_tag=1001;
    if(mpi_rank==0) {
        memcpy(global_energy, local_energy, local_phys_size*sizeof(double));
        for(l=1; l<mpi_size; l++)
            MPI_Recv(&global_energy[l*local_phys_size], local_phys_size,
                     MPI_DOUBLE, l, e_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sprintf(filename, "%s/energy_step%d.vtk", folder1, eps_tag);
        fp = fopen(filename, "w");
        if(fp) {
            fprintf(fp,"# vtk DataFile Version 3.0\n");
            fprintf(fp,"Free Energy Step %d\nASCII\n", eps_tag);
            fprintf(fp,"DATASET STRUCTURED_POINTS\n");
            fprintf(fp,"DIMENSIONS %d %d 1\n", Nx, Nz);
            fprintf(fp,"ORIGIN 0 0 0\nSPACING %f %f 1.0\n", h, h);
            fprintf(fp,"POINT_DATA %d\n", global_phys_size);
            fprintf(fp,"SCALARS free_energy float 1\nLOOKUP_TABLE default\n");
            for(int zo=0; zo<Nz; zo++)
                for(int xo=0; xo<Nx; xo++)
                    fprintf(fp,"%.6e\n", global_energy[xo*Nz+zo]);
            fclose(fp);
            printf("[Energy] Saved -> %s\n", filename);
        }
    } else {
        MPI_Send(local_energy, local_phys_size, MPI_DOUBLE, 0, e_tag, MPI_COMM_WORLD);
    }

    free(local_energy);
    if(global_energy) free(global_energy);
    return 0;
}

/* ======================================================
 * msgout
 * ====================================================== */
int msgout(double **phi, double *c_total, double **ci, double *f3,
           double h, int mpi_rank, int mpi_size,
           double msg_time, unsigned long iloop)
{
    int blocksize=Nx/mpi_size, i, z, k;
    double local_mass[NUM_COMP+2]; memset(local_mass,0,sizeof(local_mass));
    double area_elem=h*h;
    for(i=1;i<=blocksize;i++) for(z=1;z<=Nz;z++)
        for(k=1;k<=NUM_COMP;k++) if(phase_active[k]) local_mass[k]+=PHI(k,i,z)*area_elem;
    double global_mass[NUM_COMP+2]; memset(global_mass,0,sizeof(global_mass));
    for(k=1;k<=NUM_COMP;k++)
        MPI_Allreduce(&local_mass[k],&global_mass[k],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    if(mpi_rank==0) {
        printf("Time=%.4e Step=%lu | ",msg_time,iloop);
        /* 只打印活跃相 */
        for(k=1;k<=NUM_COMP;k++) if(phase_active[k]&&global_mass[k]>1e-6)
            printf("M%d=%.3f ",k,global_mass[k]);
        printf("\n");
    }
    return 0;
}

/* ======================================================
 * read_field
 * ====================================================== */
int read_field(double **phi, double **ci, double *c_total,
               double *comm_buf, double *comm_buf1,
               char *folder, int index, int mpi_rank, int mpi_size)
{
    int blocksize=Nx/mpi_size, nz_phys=Nz;
    int local_phys_size=blocksize*nz_phys;
    char filename[256]; FILE *fp;
    sprintf(filename,"%s/restart_rank_%d_step%d.bin",folder,mpi_rank,index);
    fp=fopen(filename,"rb");
    if(!fp) { fprintf(stderr,"[Rank %d] Cannot open %s\n",mpi_rank,filename); return -1; }
    double *buffer=(double*)malloc(local_phys_size*sizeof(double));
    if(!buffer) { fclose(fp); return -1; }
    for(int k=1;k<=NUM_COMP;k++) {
        if(fread(buffer,sizeof(double),local_phys_size,fp)!=(size_t)local_phys_size) {
            fprintf(stderr,"[Rank %d] Read error\n",mpi_rank);
            free(buffer); fclose(fp); return -1;
        }
        int idx=0;
        for(int i=1;i<=blocksize;i++) for(int z=1;z<=nz_phys;z++) PHI(k,i,z)=buffer[idx++];
    }
    free(buffer); fclose(fp);
    if(comm_buf&&comm_buf1) xdir_mpi_sr(phi,ci,c_total,graphic_x,graphic_z,NULL,comm_buf,comm_buf1,mpi_rank,mpi_size);
    if(mpi_rank==0) printf("Restart loaded step %d.\n",index);
    return 0;
}

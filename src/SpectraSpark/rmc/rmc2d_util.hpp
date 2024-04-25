#ifndef RMC2D_UTIL_HPP
#define RMC2D_UTIL_HPP

#include "rmc2d.hpp"
#include <fstream>
#include <iostream>

namespace RMC {

/**
 * @brief Save the configuration to a file
 * @details Save the configuration to a file in the following format:
 * comment:
 *   comment
 * La  Lb  N:
 *   La Lb N
 * a:
 *   a[0] a[1] ... a[N-1]
 * b:
 *   b[0] b[1] ... b[N-1]
 *
 * @param filename the name of the file to save
 * @param sim the simulator object
 * @return void
 */
void save_config(const std::string &filename, const RMC::Simulator2d &sim);

/**
 * @brief Load the configuration from a file
 * @note Assuming the memory for a and b is not allocated
 * @note file format is compatible with save_config
 *
 * @param filename the name of the file to load
 * @param n a pointer to save the number of particles
 * @param La a pointer to save the a dimension of the model space
 * @param Lb a pointer to save the b dimension of the model space
 * @param a array to keep the a coordinates of particles
 * @param b array to keep the b coordinates of particles
 * @return void
 *
 */
void load_config(const std::string &filename, int *n, int *La, int *Lb, int *a,
                 int *b);

/**
 * @brief Save the intensity to a file
 * @details Save the intensity to a file in the following format:
 * qx:
 *   w
 *   qx[0] qx[1] ... qx[w-1]
 * qy:
 *   h
 *   qy[0] qy[1] ... qy[h-1]
 * i:
 *   i[0][0] i[0][1] ... i[0][w-1]
 *   i[1][0] i[1][1] ... i[1][w-1]
 *   ...
 *   i[h-1][0] i[h-1][1] ... i[h-1][w-1]
 *
 * @param filename the name of the file to save
 * @param w the width of the intensity
 * @param h the height of the intensity
 * @param qx the qx values
 * @param qy the qy values
 * @param intensity the intensity values
*/
void save_i(const std::string &filename, int w, int h, double *qx, double *qy,
            double *intensity);

/**
 * @brief Load the intensity from a file
 * @note Assuming the memory for qx, qy, and intensity is not allocated
 * @note file format is compatible with save_i
 *
 * @param filename the name of the file to load
 * @param width a pointer to save the width of the intensity
 * @param height a pointer to save the height of the intensity
 * @param qx array to keep the qx values
 * @param qy array to keep the qy values
 * @param intensity array to keep the intensity values
 */
void load_i(const std::string &filename, int &width, int &height, double *&qx,
            double *&qy, double *&intensity);

/**
 * @brief Save the result to a file
 * @details save intensity, configuration after fitting and the residual history
 *
 * @param filename the name of the file to save
 * @param sim the simulator object
 * @param n_step the number of steps
 * @param res_hist the residual history
 */
void save_result(const std::string &filename, const Simulator2d &sim,
                 int n_step, double *const res_hist);

/**
 * @brief Generate the intensity on given qx, qy
 * @details intensity has periodic peak that is defined by Lx, Ly
 * @note Assuming the memory for i_exp is allocated
 *
 * @param w the width of the intensity
 * @param h the height of the intensity
 * @param qx the qx values
 * @param qy the qy values
 * @param i_exp array to keep the intensity values
 * @param Lx x interval of peaks
 * @param Ly y interval of peaks
 */
void gen_sample_data(int w, int h, double *qx, double *qy, double *i_exp,
                  double Lx, double Ly);

/**
 * @brief Generate the initial configuration
 * @details generate periodic configuration of particles
 * @param n the number of particles
 * @param La the a dimension of the model space
 * @param Lb the b dimension of the model space
 * @param da the a interval of particles
 * @param db the b interval of particles
 * @param a array to keep the a coordinates of particles
 * @param b array to keep the b coordinates of particles
 */
void gen_initial_config(int n, int La, int Lb, int da, int db,
                        int *a, int *b);

}  // namespace RMC

#endif // RMC2D_UTIL_HPP
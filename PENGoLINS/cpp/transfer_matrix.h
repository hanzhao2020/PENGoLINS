// Copyright (C) 2016 Patrick E. Farrell and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#ifndef TRANSFER_MATRIX_H
#define TRANSFER_MATRIX_H

#include <memory>
#include <vector>
#include <petscdm.h>
#include <petscvec.h>
#include <dolfin/common/MPI.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/log/log.h>

namespace dolfin
{

  class FunctionSpace;
  class BoundingBoxTree;

  /// This class builds and stores of collection of PETSc DM objects
  /// from a hierarchy of FunctionSpaces objects. The DM objects are
  /// used to construct multigrid solvers via PETSc.
  ///
  /// Warning: This classs is highly experimental and will change

  class PETScDMCollectionTemp : public PETScObject
  {
  public:

    /// Construct PETScDMCollectionTemp from a vector of
    /// FunctionSpaces. The vector of FunctionSpaces is stored from
    /// coarse to fine.
    PETScDMCollectionTemp(std::vector<std::shared_ptr<const FunctionSpace>> function_spaces);

    /// Destructor
    ~PETScDMCollectionTemp();

    /// Return the ith DM objects. The coarest DM has index 0. Use
    /// i=-1 to get the DM for the finest level, i=-2 for the DM for
    /// the second finest level, etc.
    DM get_dm(int i);

    /// These are test/debugging functions that will be removed
    void check_ref_count() const;

    /// Debugging use - to be removed
    void reset(int i);

    /// Create the interpolation matrix from the coarse to the fine
    /// space (prolongation matrix)
    static std::shared_ptr<PETScMatrix>
      create_transfer_matrix(const FunctionSpace& coarse_space,
                             const FunctionSpace& fine_space);

    static std::shared_ptr<PETScMatrix>
      create_transfer_matrix_partial_derivative(const FunctionSpace& coarse_space,
                             const FunctionSpace& fine_space, std::size_t partial_dir);

  private:

    // Find the nearest cells to points which lie outside the domain
    static void find_exterior_points(MPI_Comm mpi_comm,
                                     std::shared_ptr<const BoundingBoxTree> treec,
                                     int dim, int data_size,
                                     const std::vector<double>& send_points,
                                     const std::vector<int>& send_indices,
                                     std::vector<int>& indices,
                                     std::vector<std::size_t>& cell_ids,
                                     std::vector<double>& points);


    // Pointers to functions that are used in PETSc DM call-backs
    static PetscErrorCode create_global_vector(DM dm, Vec* vec);
    static PetscErrorCode create_interpolation(DM dmc, DM dmf, Mat *mat,
                                               Vec *vec);
    static PetscErrorCode coarsen(DM dmf, MPI_Comm comm, DM* dmc);
    static PetscErrorCode refine(DM dmc, MPI_Comm comm, DM* dmf);


    // The FunctionSpaces associated with each level, starting with
    // the coarest space
    std::vector<std::shared_ptr<const FunctionSpace>> _spaces;

    // The PETSc DM objects
    std::vector<DM> _dms;

  };

}

#endif
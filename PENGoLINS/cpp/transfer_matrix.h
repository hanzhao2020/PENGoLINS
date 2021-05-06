#ifndef TRANSFER_MATRIX_H
#define TRANSFER_MATRIX_H

#include <memory>
#include <vector>
#include <petscdm.h>
#include <petscvec.h>
#include <dolfin/common/MPI.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/log/log.h>

namespace dolfin{
  class FunctionSpace;

  class BoundingBoxTree;

  class transfer_matrix : public PETScObject{
    public:
      // Constructor
      transfer_matrix(std::vector<std::shared_ptr<const FunctionSpace>> function_spaces);

      // Destructor
      ~transfer_matrix();

      // Return the ith DM objects. The coarest DM has index 0. Use 
      // i=-1 to get the DM for the finest level, i=-2 for the DM for 
      // the second finest level, etc.
      DM get_dm(int i);

      // These are test/debugging functions that will be removed
      void check_ref_count() const;

      // Debugging use - to be removed
      void reset(int i);

      static std::shared_ptr<PETScMatrix> 
        create_transfer_matrix(const FunctionSpace& coarse_space,
          const FunctionSpace& fine_space);

      static std::shared_ptr<PETScMatrix>
        create_transfer_matrix_partial_derivative(const FunctionSpace& coarse_space,
          const FunctionSpace& fine_space, std::size_t partial_dir);

    private:
      // Find the nearest cells to points which lie outside the domain.
      static void find_exterior_points(MPI_Comm mpi_comm,
                                       std::shared_ptr<const BoundingBoxTree> treec,
                                       int dim, int data_size,
                                       const std::vector<double>& send_points,
                                       const std::vector<int>& send_indices,
                                       std::vector<int>& indices,
                                       std::vector<std::size_t>& cell_ids,
                                       std::vector<double>& points);
      //Pointers to functions that are used in PETSc DM call-backs
      static PetscErrorCode create_global_vector(DM dm, Vec* vec);
      static PetscErrorCode create_interpolation(DM dmc, DM dmf, Mat *mat,
                                                 Vec *vec);
      static PetscErrorCode coarsen(DM dmf, MPI_Comm comm, DM* dmc);
      static PetscErrorCode refine(DM dmc, MPI_Comm comm, DM* dmf);

      // The FunctionSpaces associated with each level, starting with
      // the coarest space
      std::vector<std::shared_ptr<const FunctionSpace>> _spaces;

      // The PETSc DM ojects
      std::vector<DM> _dms;
  };
}

#endif
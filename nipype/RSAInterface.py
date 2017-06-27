from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, TraitedSpec
from nipype.utils.filemanip import split_filename
import nibabel as nb
import numpy as np
import os
import itertools
from sklearn import linear_model

class RegressRSAInputSpec(BaseInterfaceInputSpec):
    # filename containing NUM_BETA_IMAGES x NUM_BETA_IMAGES x NUM_REGRESSORS np array (.npy)
    model_matrices = traits.File(exists=True,desc='RSA model matrices file',mandatory=True)
    # names of regressors
    matrix_names = traits.List(trait=traits.String, desc='names of regressors',mandatory=True)
    # list of brain volume (.nii) filenames
    volumes = traits.List(trait=traits.String, desc='functional brain volumes', mandatory=True) 
    # integer
    sphere_size = traits.Int(desc='size of searchlight sphere in voxels',mandatory=True)


class RegressRSAOutputSpec(TraitedSpec):
    similarity_maps = traits.List(trait=traits.File,desc='searchlight similarity maps')
   
class RegressRSA(BaseInterface):
    input_spec = RegressRSAInputSpec # bound to inputs attribute of SimpleInterface object
    output_spec = RegressRSAOutputSpec # bound of outputs

    def _run_interface(self, runtime): 

        def stretch_matrix(in_array):
        '''stretch a lower-triangle matrix into a numeric vector'''
        out_vector = np.empty([in_array.shape[0]**2-in_array.shape[0],1])
        i_range = range(1,in_array.shape[0])
        k=0
        for i in i_range:
            j_range = range(i)
            for j in j_range:
                out_vector[k,0]=in_array[i,j]
                k+=1

        def get_corr_tril(in_array,out_array):
            '''get correlation matrix (lower triangle)'''
            # in_array: m x n
            # out_array: m x m
            i_range = range(1,in_array.shape[0])
            for i in i_range:
                j_range = range(i)
                for j in j_range:
                    out_array[i,j]=np.correlate(in_array[i,:],in_array[j,:])
            return out_array

        def generate_sph_inds(sph_radius):
            '''generate the voxel offsets within a sphere of given radius'''
            # array of potential shift values
            v = range(-(sph_radius-2),sph_radius-1)
            # all the potential non-edge shifts around the center voxel
            vindices_1=list(itertools.product(range(len(v)),repeat=3))
            # all the potential shifts that can occur next to an edge voxel
            vindices_2=list(itertools.product(range(len(v)),repeat=2))
            # associate each shift with its shift value from v
            c1=np.empty([len(vindices_1),3])
            for i in range(len(vindices_1)):
                c1[i,:] = [v[vindices_1[i][0]],v[vindices_1[i][1]],v[vindices_1[i][2]]]
            c2=np.empty([len(vindices_2),2])
            for i in range(len(vindices_2)):
                c2[i,:] = [v[vindices_2[i][0]],v[vindices_2[i][1]]]
            # shifts for voxels at the exterior of the sphere (that poke out)
            edges=np.array([[sph_radius-1 for i in range(len(c2))]])
            c3 = np.concatenate((edges.T,c2),axis=1)
            c4 = np.concatenate((-edges.T,c2),axis=1)
            c5 = np.concatenate((c2,edges.T),axis=1)
            c6 = np.concatenate((c2,-edges.T),axis=1)
            c7 = np.concatenate((np.array([c2[:,0]]).T,edges.T,np.array([c2[:,1]]).T),axis=1)
            c8 = np.concatenate((np.array([c2[:,0]]).T,-edges.T,np.array([c2[:,1]]).T),axis=1)
            coords=np.concatenate((c1,c3,c4,c5,c6,c7,c8),axis=0)
            return coords

        # read in brain volumes
        # (assume grey-matter masking or other masking has already happened)
        brain_size = []
        for each_vol in range(len(self.inputs.volumes)):
            brain_vol = nb.load(self.inputs.volumes[each_vol]) 
            if not brain_size:
                brain_size = brain_vol.shape
                brain_data = np.empty([brain_size[0],brain_size[1],brain_size[2],len(self.inputs.volumes)+1])
            brain_data[:,:,:,each_vol] = brain_vol.get_data() #assumes 3D images (i.e., last dim is 1)
        # read in regressor matrices 
        matrices = np.load(self.inputs.model_matrices)
        matrix_arr = np.empty([(matrices.shape[0]**2 - matrices.shape[0]),matrices.shape[2]])
        for i in range(matrices.shape[2]):
            matrix_arr[:,i]=stretch_matrix(matrices[:,:,i])
        # define locations of voxels in sphere
        sphere_size = self.inputs.sphere_size
        # NUM_VOXELS_IN_SPHERE x 3 array
        voxel_offsets = generate_sph_inds(sphere_size)
        size_of_sph = voxel_offsets.shape[0]
        # array for regressor weights (BRAIN_X x BRAIN_Y x BRAIN_Z x NUM_REGRESSORS+1)
        map_weights = np.zeros([brain_size[0],brain_size[1],brain_size[2],matrices.shape[2]+1])
        # array for voxel values (NUM_BETA_IMAGES x NUM_VOXELS_IN_SPHERE)
        sph_voxels = np.empty([matrices.shape[0],size_of_sph])
        sph_voxels.fill(np.nan)
        # array for neural similarity matrices (NUM_BETA_IMAGES x NUM_BETA_IMAGES)
        sphere_corrs = np.zeros([matrices.shape[0]),matrices.shape[0]])

        # for each sphere in brain:
        for (i,j,k) in itertools.product(range(brain_size[0]),range(brain_size[1]),range(brain_size[2])):
            row_counter += 1
            # define the sphere indices: NUM_VOXELS_IN_SPHERE x 3 array
            voxel_inds = voxel_offsets + [i,j,k]
            for voxind in range(voxel_inds.shape[0]): # for each triplet of indices
                # insert a column for that voxel's data from all beta images
                sph_voxels[:,voxind]=brain_data[voxel_inds[voxind,0],voxel_inds[voxind,1],voxel_inds[voxind,2],:]
            # check voxels in sphere for quality
            if len(np.where(np.isnan(sph_voxels))) >= 0.5*(sph_voxels.shape[0]*sph_voxels.shape[1]):
            # if more than half of the voxels in the sphere, across all beta images, have values
                # create neural similarity matrix
                sphere_corrs = get_corr_tril(sph_voxels,sphere_corrs)
                # run your model in the neural similarity space
                rsa_model = linear_model.LinearRegression()
                rsa_model.fit(matrices,sphere_corrs)
                # write out similarity weights to brain matrix (1 value per voxel/regressor)
                map_weights[i,j,k,:] = rsa_model.coef_
        # write out similarity maps
        for m in range(len(matrix_names)):
            out_name = matrix_names[m] + '.nii' 
            out_map = nb.Nifti1Image(map_weights[:,:,:,m],np.eye(3)) 
            nb.save(out_map,out_name)
        
        return runtime

    def _list_outputs(self):
        # outputs is a dictionary object
        # keys correspond to OutputSpec attributes
        outputs = self._outputs().get()
        # list of similarity map filenames
        outputs["similarity_maps"] = list([nm + '.nii' for nm in self.inputs.matrix_names])
        return outputs

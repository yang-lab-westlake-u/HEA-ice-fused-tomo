support_para = struct('th_dis_r_afterav', 0.90, 'dilate_size', 15, ...
'erode_size', 15, 'bw_size', 5000);
tight_support = obtain_tight_support(chem_total,support_para);
se = strel3d(10);
tight_support2 = imdilate(tight_support,se);

element_names = {'Au', 'Ag', 'Bi', 'Co', 'Cu', 'In', 'Ni', 'Pd', 'Pt', 'Rh', 'Ru'};

hdf5_file = 'fusion_recon.h5';

for i = 1:length(element_names)
    dataset_name = ['/' element_names{i}];

    element_data = h5read(hdf5_file, dataset_name);
end

for i = 1:length(element_names)
    dataset_name = ['/' element_names{i}];
    
    element_data = h5read(hdf5_file, dataset_name);
    
    element_mask = element_data .* tight_support2;
    
    save_filename = [element_names{i} '.mat'];
    mask_var_name = [element_names{i} '_mask'];
    
    eval([mask_var_name ' = element_mask;']); 
    save(save_filename, mask_var_name);      
end
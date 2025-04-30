file_name = 'fusion_recon.h5';
element_names = {'Au', 'Ag', 'Bi', 'Co', 'Cu', 'In', 'Ni', 'Pd', 'Pt', 'Rh', 'Ru'};

chem_total = zeros(320, 320, 320);

for i = 1:length(element_names)
    dataset_name = ['/' element_names{i}]; 
    chem_map = h5read(file_name, dataset_name);  
    chem_total = chem_total + chem_map;
end


save('chem_total.mat', 'chem_total');
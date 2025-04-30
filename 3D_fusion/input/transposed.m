% Define the list of elements to process
elements = {'Au', 'Ag', 'Bi', 'Co', 'Cu', 'In', 'Ni', 'Pd', 'Pt', 'Rh', 'Ru'};

% Loop through each element
for i = 1:length(elements)
    % Current element name
    element = elements{i};
    
    % Construct input filename and variable name
    input_filename = sprintf('projs_fused_%s.mat', element);
    variable_name = sprintf('fused_%s', element); % Assuming the variable name is fused_element
    
    % Check if the input file exists
    if isfile(input_filename)
        fprintf('Processing file: %s\n', input_filename);
        
        % Load file content
        data = load(input_filename);
        
        % Check if the specified variable is present
        if isfield(data, variable_name)
            % Extract the variable
            fused_data = data.(variable_name);
            
            % Transpose the data
            fused_data = permute(fused_data, [2, 1, 3]); % Swap the first two dimensions
            
            % Construct output filename
            output_filename = sprintf('transposed_projs_fused_%s.mat', element);
            
            % Save using a dynamic variable name
            eval(sprintf('%s = fused_data;', variable_name)); % Create dynamic variable
            save(output_filename, variable_name); % Save the dynamic variable
            fprintf('Transposed file saved: %s\n', output_filename);
        else
            fprintf('Variable %s not found in file %s, skipping.\n', variable_name, input_filename);
        end
    else
        fprintf('File %s does not exist, skipping.\n', input_filename);
    end
end

fprintf('Processing of all elements completed!\n');
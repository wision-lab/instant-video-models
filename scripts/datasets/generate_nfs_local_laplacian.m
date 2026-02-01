% We assume that the pwd is the directory containing the script
working_dirpath = pwd();

% Remove "scripts/datasets" from the end of the path
project_dirpath = working_dirpath(1 : end - 16);

input_base_dirpath = project_dirpath + "data/nfs";

% Uncomment these lines for the "strong" version
% Parameter values were pulled from the detail manipulation demo
% (lapfilter_demo.m) in the published source code from Paris et al 2011
% https://people.csail.mit.edu/sparis/publi/2011/siggraph/matlab_source_code.zip
% ==============================================================================
% sigma = 0.4;
% alpha = 0.25;
% output_base_dirpath = project_dirpath + "data/nfs_laplacian_strong";
% ==============================================================================

% Uncomment these lines for the "moderate" version
% Bring the value of alpha closer to 1 to reduce detail enhancement intensity
% ==============================================================================
sigma = 0.4;
alpha = 0.5;
output_base_dirpath = project_dirpath + "data/nfs_laplacian_moderate";
% ==============================================================================

jpg_files = dir(input_base_dirpath + "/**/*.jpg");
n = length(jpg_files);
for i = 1 : n
    fprintf("%d of %d (%.2f%%)\r", i, n, i / n * 100.0)
    input_filepath = char(jpg_files(i).folder + "/" + jpg_files(i).name);

    % Replace the leading portion of the input filepath with the output dirpath
    filepath_tail = input_filepath(strlength(input_base_dirpath) + 1 : end);
    output_filepath = output_base_dirpath + string(filepath_tail);

    % Skip already processed items
    if exist(output_filepath, "file")
        continue
    end

    % Create the output directory if it doesn't already exist
    [output_dirpath, name, ext] = fileparts(output_filepath);
    if ~exist(output_dirpath, "dir")
        mkdir(output_dirpath)
    end

    % Apply the locallapfilt filter
    input_image = imread(input_filepath);
    output_image = locallapfilt(input_image, sigma, alpha);

    % Use the default jpeg image quality of 75
    imwrite(output_image, output_filepath)
end

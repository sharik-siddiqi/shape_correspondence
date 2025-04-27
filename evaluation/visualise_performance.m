clear all;

% Load the geodesic error curves and performance data
load('<path_to_curve_geo_error.mat>'); % Placeholder for geodesic error curves and performance data

% Increment face indices to match MATLAB 1-based indexing
faces = faces + 1;

% Initialize an array to store the area under the curve (AUC) for each match
area = [];

% Calculate the AUC for each match and store it
for i = 1:size(match, 1)
    % Compute the error curve for the current match
    x = compute_error_curve(geo_err(i, :), thr);
    
    % Compute the area under the curve and append it to the area array
    area = [area, trapz(thr, x)];
end

% Display the mean area under the curve
disp(['Mean AUC: ', num2str(mean(area))]);

% Sort the matches based on their AUC in ascending order
[~, sorted_indices] = sort(area, 'ascend');

% Visualize the matches
for i = sorted_indices
    % Get the indices of the source and target shapes for the current match
    num_1 = source(1, i) + 1;
    num_2 = target(1, i) + 1;

    % Extract the vertex and face data for the source and target shapes
    shape_1.VERT = squeeze(vertices(num_1, :, :));
    shape_2.VERT = squeeze(vertices(num_2, :, :));
    shape_1.TRIV = faces;
    shape_2.TRIV = faces;

    % Set the number of vertices for each shape
    shape_1.n = size(shape_1.VERT, 1);
    shape_2.n = size(shape_2.VERT, 1);

    % Extract the matches for the supervised case
    match_this_sup_1 = match(i, :);
    match_this_sup_2 = match(find(source == target(1, i) & target == source(1, i)), :);

    % Plot the 3D correspondences for the supervised case
    cmap_sup = plot_3Dcorrespondence(shape_1, shape_2, ...
        [[1:size(match, 2)]', (match_this_sup_1 + 1)'], ...
        [[1:size(match, 2)]', (match_this_sup_2 + 1)']);
    
    % Pause to allow visualization
    pause;
end


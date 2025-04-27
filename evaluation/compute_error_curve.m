% Function to calculate the geodesic error curve based on given errors and thresholds
% 
% Args:
%   errors: A vector containing error values for a dataset.
%   thresholds: A vector of threshold values to evaluate the error curve.
%
% Returns:
%   curve: A vector containing the percentage of errors below each threshold.

function curve = compute_error_curve(errors, thresholds)
    % Initialize the curve vector with zeros
    curve = zeros(1, length(thresholds));
    
    % Loop through each threshold
    for i = 1:length(thresholds)
        % Calculate the percentage of errors below the current threshold
        curve(i) = 100 * sum(errors <= thresholds(i)) / length(errors);
    end
end

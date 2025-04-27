% Enable software rendering for OpenGL (optional, depending on your system)
% opengl software

% Load geodesic error curves for unsupervised and supervised cases
curve_unsup = load('<path_to_unsupervised_geodesic_error_matrix.mat>'); % Placeholder for unsupervised geodesic error curves
curve_sup = load('<path_to_supervised_geodesic_error_matrix.mat>');     % Placeholder for supervised geodesic error curves

% Calculate the mean geodesic error per vertex on the test dataset
mean_unsup = mean(curve_unsup.mean_curves(2, :)); % Mean error for unsupervised case
mean_sup = mean(curve_sup.mean_curves(2, :));     % Mean error for supervised case

% Display the mean errors
disp(['Mean geodesic error (Unsupervised): ', num2str(mean_unsup)]);
disp(['Mean geodesic error (Supervised): ', num2str(mean_sup)]);

% Plot geodesic error curves for supervised and unsupervised cases
figure;
plot(curve_unsup.thr, curve_unsup.mean_curves(2, :), 'LineWidth', 2, 'DisplayName', 'Unsupervised');
hold on;
plot(curve_sup.thr, curve_sup.mean_curves(2, :), 'LineWidth', 2, 'DisplayName', 'Supervised');
grid on;
legend('show');
xlabel('Threshold');
ylabel('Geodesic Error (%)');
title('Geodesic Error Curves (Supervised vs Unsupervised)');
hold off;

% Calculate the area under the curve (AUC) for both cases
for i = 1:2
    auc_unsup = trapz(curve_unsup.thr, curve_unsup.mean_curves(i, :)); % AUC for unsupervised case
    auc_sup = trapz(curve_sup.thr, curve_sup.mean_curves(i, :));       % AUC for supervised case
    
    % Display the AUC results
    disp(['AUC (Unsupervised, Curve ', num2str(i), '): ', num2str(auc_unsup)]);
    disp(['AUC (Supervised, Curve ', num2str(i), '): ', num2str(auc_sup)]);
end

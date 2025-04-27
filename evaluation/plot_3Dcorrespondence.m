function [cmapN] = plot_3Dcorrespondence(M, N, corr, corr_2)
    % Plot 3D correspondences between two shapes M and N.
    %
    % Args:
    %   M: A structure representing the source shape with fields:
    %      - VERT: Vertex coordinates (n x 3).
    %      - TRIV: Triangular faces (m x 3).
    %      - n: Number of vertices.
    %   N: A structure representing the target shape with fields:
    %      - VERT: Vertex coordinates (n x 3).
    %      - TRIV: Triangular faces (m x 3).
    %      - n: Number of vertices.
    %   corr: Correspondence mapping from M to N (n x 2).
    %   corr_2: Backup correspondence mapping for unmatched vertices (n x 2).
    %
    % Returns:
    %   cmapN: A colormap for the target shape N.

    % Create a figure for visualization
    figure;

    % Generate a colormap for the source shape M
    cmapM = create_colormap(M, M);

    % Plot the source shape M with its colormap
    subplot(121);
    trisurf(M.TRIV, M.VERT(:, 1), M.VERT(:, 2), M.VERT(:, 3), 1:M.n, 'EdgeAlpha', 0);
    axis equal;
    axis off;
    colormap(gca, cmapM);

    % Initialize the colormap for the target shape N
    cmapN = zeros(N.n, 3);

    % Map the colormap from M to N based on the correspondences
    cmapN(corr(:, 2), :) = cmapM(corr(:, 1), :);

    % Handle unmatched vertices in N using the backup correspondence
    for i = 1:size(cmapN, 1)
        if all(cmapN(i, :) == 0)  % Check if the vertex has no assigned color
            cmapN(i, :) = cmapM(corr_2(i, 2), :);
        end
    end

    % Plot the target shape N with its colormap
    subplot(122);
    trisurf(N.TRIV, N.VERT(:, 1), N.VERT(:, 2), N.VERT(:, 3), 1:N.n, 'EdgeAlpha', 0);
    axis equal;
    axis off;
    colormap(gca, cmapN);
end

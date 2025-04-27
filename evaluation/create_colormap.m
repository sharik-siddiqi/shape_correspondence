function colors = compute_vertex_colormap(M, N)
    % Compute a colormap for the vertices of mesh M based on their normalized coordinates.
    %
    % Args:
    %   M: A structure representing the first mesh, containing a field `VERT` (vertices).
    %   N: A structure representing the second mesh, containing a field `VERT` (vertices).
    %
    % Returns:
    %   colors: A matrix where each row represents the RGB color for a vertex in M.

    % Find the minimum and maximum coordinates across both meshes
    minx = min(min(M.VERT(:, 1)), min(N.VERT(:, 1)));
    miny = min(min(M.VERT(:, 2)), min(N.VERT(:, 2)));
    minz = min(min(M.VERT(:, 3)), min(N.VERT(:, 3)));
    maxx = max(max(M.VERT(:, 1)), max(N.VERT(:, 1)));
    maxy = max(max(M.VERT(:, 2)), max(N.VERT(:, 2)));
    maxz = max(max(M.VERT(:, 3)), max(N.VERT(:, 3)));

    % Normalize the vertex coordinates of M to the range [0, 1]
    colors = [
        (M.VERT(:, 1) - minx) / (maxx - minx), ...
        (M.VERT(:, 2) - miny) / (maxy - miny), ...
        (M.VERT(:, 3) - minz) / (maxz - minz)
    ];
end

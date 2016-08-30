function resI = generateProjections(I, theta_, phi_, ksi_)
    % Create inline function RotMat(theta, ksi, phi)
    RotMat = @(theta, ksi, phi) [cos(theta)*cos(ksi) (-cos(phi)*sin(ksi) + sin(phi)*sin(theta)*cos(ksi)) (sin(phi)*sin(ksi) + cos(phi)*sin(theta)*cos(ksi));...
            cos(theta)*sin(ksi) (cos(phi)*cos(ksi) + sin(phi)*sin(theta)*sin(ksi)) (-sin(phi)*cos(ksi) + cos(phi)*sin(theta)*sin(ksi));...
       -sin(theta) sin(phi)*cos(theta) cos(phi)*cos(theta)];

    %If you want to make deformation stronger, reduce foc_len
    foc_len = (size(I,1 ) + size(I,2))/5; 
    %Initialize camera matrix
    CamMat = [foc_len 0 size(I,1)/2; 0 foc_len size(I,2)/2; 0 0 1];
    counter = 1;
    for t = 1:length(theta_)
        for p = 1:length(phi_)
            for k = 1:length(ksi_)
                CamMat = [foc_len 0 size(I,1)/2; 0 foc_len size(I,2)/2; 0 0 1];
                H = CamMat * RotMat(theta_(t)*pi/180, ksi_(k)*pi/180, phi_(p)*pi/180)';
                [newim, newT] = imTrans(I, inv(H), [], max(size(I)));
                resI(:,:,counter) = imresize(newim, [size(I,1) size(I,2)]);
                %figure, imshow((resI));
                counter = counter + 1;
            end
        end
    end

end
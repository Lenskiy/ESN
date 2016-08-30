function [P T] = loadTrainingImages(path_to_images, goal_vec, format)
    % format either 'png' or 'jpg'
    cd(path_to_images)
    file_list = dir;
    clear P T
    for i = 1:size(file_list,1)
        if(findstr( file_list(i).name, format) ~= 0)
            gray = rgb2gray(imread([path_to_images '\' file_list(i).name]));
            if (~exist('P'))
                T = goal_vec';
                P = gray(:);
            else
                T = [T goal_vec'];
                P = [P gray(:)];
            end
        end
    end
end
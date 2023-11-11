%% MA
% This file is for nirs-data classification
% Most of MATLAB functions are available in BBCI toolbox
% Some minor code modifications might be applied
% We do not guarantee all of functions works properly in your platform
% If you want to see more tutorials, visit BBCI toolbox (https://github.com/bbci/bbci_public)
% Modified by Zenghui Wang, November 8, 2023.

% specify your nirs data directory (NirsMyDataDir), temporary directory (TemDir), working directory (WorkingDir), and preprocessed data (MySaveDir)
WorkingDir = pwd;
MyToolboxDir = fullfile(WorkingDir, 'bbci_public-master');
NirsMyDataDir = fullfile(WorkingDir, 'NIRS');
TemDir = fullfile(WorkingDir, 'temp');
MySaveDir = fullfile(WorkingDir, 'MA_fNIRS_data');
cd(MyToolboxDir);
startup_bbci_toolbox('DataDir', NirsMyDataDir, 'TmpDir', TemDir, 'History', 0);


for idx = 1 : 29 
    filename = num2str(idx);
    if idx <= 9
        file = strcat('subject 0', filename);
    else
        file = strcat('subject',32, filename);
    end
    disp(file);
    
    % subject
    subdir_list = {file}; 
    stimDef.nirs = {1,2; 'condition1','condition2'};

    % load nirs data
    loadDir = fullfile(NirsMyDataDir, subdir_list{1});
    cd(loadDir);
    load cnt; load mrk, load mnt; % load continous signal (cnt), marker (mrk) and montage (mnt)
    cd(WorkingDir)

    % merge cnts in each session, for mental arithmetic: ment
    cnt_temp = cnt; mrk_temp = mrk; % save data temporarily
    clear cnt mrk;

    [cnt.ment, mrk.ment] = proc_appendCnt({cnt_temp{2}, cnt_temp{4}, cnt_temp{6}}, {mrk_temp{2}, mrk_temp{4}, mrk_temp{6}}); % merged mental arithmetic cnts

    % MBLL
    cnt.ment = proc_BeerLambert(cnt.ment);

    % band-pass filtering
    [b, a] = butter(3, [0.01 0.1]/cnt.ment.fs*2);
    cnt.ment = proc_filtfilt(cnt.ment, b, a);

    % divide into HbR and HbO, cntHb uses same structure with cnt
    cntHb.ment.oxy   = cnt.ment; 
    cntHb.ment.deoxy = cnt.ment; 

    % replace data
    cntHb.ment.oxy.x = cnt.ment.x(:, 1:end/2); 
    cntHb.ment.oxy.clab = cnt.ment.clab(:, 1:end/2);
    cntHb.ment.oxy.clab = strrep(cntHb.ment.oxy.clab, 'oxy', ''); % delete 'oxy' in clab
    cntHb.ment.oxy.signal = 'NIRS (oxy)';

    cntHb.ment.deoxy.x = cnt.ment.x(:, end/2+1:end); 
    cntHb.ment.deoxy.clab = cnt.ment.clab(:, end/2+1:end);
    cntHb.ment.deoxy.clab = strrep(cntHb.ment.deoxy.clab, 'deoxy', ''); % delete 'deoxy' in clab
    cntHb.ment.deoxy.signal = 'NIRS (deoxy)'; 

    % epoching
    ival_epo = [-10 25] * 1000; % from -10000 to 25000 msec relative to task onset (0 s)
    epo.ment.oxy   = proc_segmentation(cntHb.ment.oxy, mrk.ment, ival_epo);
    epo.ment.deoxy = proc_segmentation(cntHb.ment.deoxy, mrk.ment, ival_epo);

    % baseline correction
    ival_base = [-5 -2] * 1000;
    epo.ment.oxy   = proc_baseline(epo.ment.oxy, ival_base);
    epo.ment.deoxy = proc_baseline(epo.ment.deoxy, ival_base);

    % using moving time windows
    StepSize = 1 * 1000; % msec
    WindowSize = 3 * 1000; % msec
    ival_start = (ival_epo(1):StepSize:ival_epo(end)-WindowSize)';
    ival_end = ival_start+WindowSize;
    ival = [ival_start, ival_end];
    nStep = length(ival);

    for stepIdx = 1:nStep
        segment.ment.deoxy{stepIdx} = proc_selectIval(epo.ment.deoxy, ival(stepIdx,:));
        segment.ment.oxy{stepIdx}   = proc_selectIval(epo.ment.oxy,   ival(stepIdx,:));
    end
    
    % save fNIRS data
    num = num2str(idx);
    mkdir(strcat(MySaveDir, '\',num));
    for stepIdx = 1:nStep
        path = strcat(MySaveDir, '\', num , '\', num2str(stepIdx), '_deoxy.mat');
        signal = segment.ment.deoxy{stepIdx}.x;
        save(path, 'signal');
        
        path = strcat(MySaveDir, '\', num , '\', num2str(stepIdx), '_oxy.mat');
        signal = segment.ment.oxy{stepIdx}.x;
        save(path, 'signal');
    end
    path = strcat(MySaveDir, '\', num , '\', num, '_', 'desc');
    label = segment.ment.deoxy{1}.event.desc;
    save(path, 'label');
    
    disp('MA data finish');
end


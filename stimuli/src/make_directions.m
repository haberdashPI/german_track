
function [directions,critical_times] = make_directions(config,block_cfg,trial,audiodata)
    section_lengths = block_cfg.trial_section_lengths{trial};
    switch_num = block_cfg.trial_switch_num(trial);
    switch_len = config.switch_len;
    min_stay_len = config.min_stay_len;

    idxs = block_cfg.trial_sentences(trial,:);
    s1 = audiodata{1}(idxs(1)).data;
    s2 = audiodata{2}(idxs(2)).data;
    s3 = audiodata{3}(idxs(3)).data;
    [len_stim,s1,s2,s3] = equalize_lengths(s1,s2,s3);

    % put the opening section
    switch_wave = (0:1/(config.fs*switch_len):1)';

    % then put everything together
    % (comment: I don't love the lack of modularity, but I don't need to change
    % this, so I'm leaving it alone for now)
    dir1 = [];
    dir2 = [];
    dir3 = [];

    A = 1/5;
    critical_times = [];
    sec1;
    sec2;
    sec3;
    secs_equalize;
    critical_times = cumsum(critical_times);
    to_angle;

    % send results to return values
    directions = {dir1; dir2; dir2};

    function sec1
        dir1 = make_jitter(round((section_lengths(1)+min_stay_len)*config.fs),0);
        dir2 = make_jitter(length(dir1),1);
        sl = min(length(dir2),round(length(switch_wave)/2));
        dir3 = flip([switch_wave(1:sl); make_jitter(length(dir1)-sl,1)-switch_wave(sl)]);
        critical_times = [critical_times round((section_lengths(1)+min_stay_len)*config.fs)];
    end

    function sec2
        for section_idx=2:switch_num
            if dir1(end)<0.5
                this_switch = switch_wave;
                this_len = round((section_lengths(section_idx)+switch_len*2)*config.fs);
                this_len2 = this_len-length(switch_wave)*2;
                dir2 = [dir2; make_jitter(length(this_switch),1,1); flip(switch_wave); make_jitter(this_len2,0,1); switch_wave];
                critical_times = [critical_times length(this_switch) length(switch_wave) this_len2 length(switch_wave)];
                mini_switch = switch_wave(1:ceil(this_len2/2));
                dir3 = [dir3; make_jitter(length(this_switch)*2,0); mini_switch; flip(mini_switch(1:end-mod(this_len2,2))); make_jitter(length(switch_wave),0)];
            else
                this_switch = flip(switch_wave);
                this_len = round((section_lengths(section_idx)+min_stay_len)*config.fs);
                dir2 = [dir2; make_jitter(length(switch_wave)+this_len,1)];
                mini_switch = switch_wave(1:ceil(this_len/2));
                dir3 = [dir3; make_jitter(length(switch_wave),0); mini_switch; flip(mini_switch(1:end-mod(this_len,2)))];
                critical_times = [critical_times length(this_switch) this_len];
            end
            dir1 = [dir1; this_switch; make_jitter(this_len,this_switch(end))];
        end
    end

    function sec3
        section_idx = switch_num + 1;
        if dir1(end)<0.5
            this_switch = switch_wave;
            this_len = round((section_lengths(section_idx))*config.fs);
            dir2 = [dir2; make_jitter(length(this_switch),1); flip(switch_wave(end+1-min(this_len,length(switch_wave)):end)); make_jitter(this_len-length(switch_wave),0)];
            critical_times = [critical_times length(this_switch) length(switch_wave)];
        else
            this_switch = flip(switch_wave);
            this_len = round((section_lengths(section_idx))*config.fs);
            dir2 = [dir2; make_jitter(length(switch_wave)+this_len,1)];
            critical_times = [critical_times length(this_switch) this_len];
        end
        dir1 = [dir1; this_switch; make_jitter(this_len,this_switch(end))];
        dir3 = [dir3; make_jitter(length(this_switch)+this_len,0)];
    end

    function secs_equalize
        if ~isempty(find(diff([length(dir1) length(dir2) length(dir3)])~=0,1))
            disp('dir lengths not the same!');
        end
        dir1 = dir1(1:len_stim);
        dir2 = dir2(1:len_stim);
        dir3 = dir3(1:len_stim);
    end

    function jit = make_jitter(len_requested,direc,override)
        if nargin==2, override=0; end
        jit = -sin(2*pi/(config.jitter_period)*...
            linspace(0,len_requested/config.fs,len_requested))'*A;
        if direc==1 && override
            jit = jit+direc;
        elseif direc==1 || override
            jit = -jit+direc;
        end
    end

    function to_angle
        dir1 = (dir1+A)/(1+A*2)*180-90;
        dir2 = (dir2+A)/(1+A*2)*180-90;
        dir3 = (dir3+A)/(1+A*2)*180-90;
    end
end

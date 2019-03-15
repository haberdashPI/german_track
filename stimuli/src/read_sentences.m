
function [audiodata,fs] = read_sentences(sentence_dir,ppl_order)
    reader_index = containers.Map;
    for i = 1:length(ppl_order)
        reader_index(ppl_order(i)) = i;
    end

    warning('Assuming all audio files have the same sample rate.');

    audiodata = {[], [], []};
    all_sentence_files = sort(dir(fullfile(sentence_dir,'*.wav')));
    for file_idx=1:length(all_sentence_files)
        file_name = all_sentence_files(file_idx).name;
        [data,fs] = audioread(fullfile(sentence_dir,file_name));

        sentence = [];
        sentence.data = data;
        sentence.length_s = length(passage)/fs;
        sentence.filename = file_name;

        reader_id = file_name(1:5);
        if isKey(reader_index,reader_id)
            i = reader_index(reader_id);
            audiodata{i} = ...
                [audiodata{reader_index(reader_id)}; sentence];
        else
            error(['Could not find key for file prefix: ' reader_id])
        end
    end
end

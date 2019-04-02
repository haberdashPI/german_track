library(rjson)

if(!file.exists("install.json")){
    error("Could not find install.json: see README.md")
}else if(!file.exists("data")){
    data_dir = fromJSON(file = "install.json")$data
    file.symlink(data_dir,"data")
    message(paste0("The folder `data` links to ",data_dir))
}else{
    message("The folder `data` already exists.")
}

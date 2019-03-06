pcpsp_file = strcat('newman1','.pcpsp');
fid = fopen(strcat('../minelib_inputs/',pcpsp_file));
while True
   linea =  fgetl(fid);
   linea_cell = strsplit(linea, ' ');
   switch linea_cell[0]
       
end
fclose(fid);
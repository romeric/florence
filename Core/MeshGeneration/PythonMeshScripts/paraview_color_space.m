function [] = paraview_color_space(scheme)
  if (exist('scheme') ~= 1), scheme='jet'; end
  colors = eval(scheme);
  N = length(colors);
  fid = fopen([scheme '.xml'], 'w');
  fprintf(fid, '<ColorMap name="%s" space="HSV">\n', scheme);
  for i=1:N
      x = [(i-1)/(N-1); colors(i,:)'];
      fprintf(fid, '  <Point x="%f" o="1" r="%f" g="%f" b="%f"/>\n', x);
  end
  fwrite(fid, '</ColorMap>');
  fclose(fid);
end
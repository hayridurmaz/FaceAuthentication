cd $1
images=$(for i in `ls *.jpg`; do LEN=`expr length $i`; echo  $i; done | sort -n)
j=1
for i in $images; do
  new=$(printf "%04d.jpg" "$j") #04 pad to length of 4
  mv -i -- "$i" "$new"
  let j=j+1
   #we can add a condition on j to rename just the first 999 images.
done

if [[ ! -z $2 ]]
then
  ffmpeg -framerate 25 -i %04d.jpg   -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p $2.mp4
fi
#/bin/sh

dns="domains.txt"
counter=0
group=0
# Fix to not polite folders outside the project
mpath="$HOME/data"

if [ ! -d "$mpath/g0" ]; then
	echo "Creating first group"
	mkdir -p "$mpath/g0"
fi

while IFS= read -r line
do
 file="$mpath/g$group/d$counter.txt"

 #echo "$file"
 curl -L --silent $line 1>$file &
 
 counter=$(expr $counter + 1)
 if [ $counter -eq 1000 ]; then 
	counter=0
	group=$(expr $group + 1)
	if [ ! -d "$mpath/g$group" ]; then
		mkdir "$mpath/g$group"
	fi
 fi

done < "$dns"

wait
print "Done!\n"
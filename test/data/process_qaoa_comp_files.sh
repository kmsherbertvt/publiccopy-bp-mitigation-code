for dir in ./adapt_qaoa_comp/*/; do
    cat $dir/*.py | grep "elist =" | sed "s/elist = //g" > $dir/elist.txt
    cat $dir/error*.txt | sed "s/[0-9]         //g;s/\n/, /g" > $dir/error_formatted.txt
done
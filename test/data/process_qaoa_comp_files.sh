for dir in ./adapt_qaoa_comp/*/; do
    cat $dir/*.py | grep "elist =" | sed "s/elist = //g" > $dir/elist.txt
    cat $dir/error*.txt | sed -z "s/[0-9]         //g;s/\n/, /g" | sed 's/^/[/;s/$/]/' > $dir/error_formatted.txt
done
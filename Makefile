TEX=pdflatex

rapport.pdf: doc/rapport.tex 
	$(TEX) -output-directory doc/ doc/rapport.tex

clean:
	rm doc/*.aux doc/*.log doc/*latexmk doc/*.gz doc/*.f*
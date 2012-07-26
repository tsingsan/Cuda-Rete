all:  
	+@$(MAKE) -C ./ReteEngine
	+@$(MAKE) -C ./Sample

run:all
	+@$(MAKE) -C ./Sample run

clean: 
	+@$(MAKE) -C ./Sample clean
	+@$(MAKE) -C ./ReteEngine clean

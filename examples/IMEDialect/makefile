#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
BUDDY_LLC := ../../build/bin/buddy-llc

.DEFAULT_GOAL := all
PYTHON ?= python3
A100_OUT ?= ../../build/ime-a100
A100_LOWER_FLAGS = -lower-linalg-to-ime="target=k3" -lower-ime="target=k3" \
	-expand-strided-metadata -convert-scf-to-cf -convert-cf-to-llvm \
	-convert-arith-to-llvm -convert-func-to-llvm \
	-finalize-memref-to-llvm -reconcile-unrealized-casts

.PHONY: a100-ssa-object k3-dot-object
a100-ssa-object:
	$(PYTHON) generate_a100_tests.py $(A100_OUT)
	$(BUDDY_TRANSLATE) -buddy-to-llvmir $(A100_OUT)/a100-kernels.mlir \
		-o $(A100_OUT)/a100-kernels.ll
	$(BUDDY_LLC) -mtriple=riscv64 -mcpu=spacemit-a100 -verify-machineinstrs \
		-filetype=obj $(A100_OUT)/a100-kernels.ll -o $(A100_OUT)/a100-kernels.o

k3-dot-object:
	mkdir -p $(A100_OUT)
	$(BUDDY_OPT) k3_dot_test.mlir $(A100_LOWER_FLAGS) \
		-o $(A100_OUT)/k3-dot.mlir
	$(BUDDY_TRANSLATE) -buddy-to-llvmir $(A100_OUT)/k3-dot.mlir \
		-o $(A100_OUT)/k3-dot.ll
	$(BUDDY_LLC) -mtriple=riscv64 -mcpu=spacemit-a100 -verify-machineinstrs \
		-filetype=obj $(A100_OUT)/k3-dot.ll -o $(A100_OUT)/k3-dot.o

.PHONY: all clean vmadot vmadotu vmadotsu vmadotus help \
        vmadot-lower vmadot-translate vmadot-asm vmadot-run \
        vmadotu-lower vmadotu-translate vmadotu-asm vmadotu-run \
        vmadotsu-lower vmadotsu-translate vmadotsu-asm vmadotsu-run \
        vmadotus-lower vmadotus-translate vmadotus-asm vmadotus-run \
        vmadot1-lower vmadot1-translate vmadot1-asm vmadot1-run \
        vmadot1u-lower vmadot1u-translate vmadot1u-asm \
        vmadot1su-lower vmadot1su-translate vmadot1su-asm \
        vmadot1us-lower vmadot1us-translate vmadot1us-asm \
        vmadot2-lower vmadot2-translate vmadot2-asm vmadot2-run \
        vmadot2u-lower vmadot2u-translate vmadot2u-asm \
        vmadot2su-lower vmadot2su-translate vmadot2su-asm \
        vmadot2us-lower vmadot2us-translate vmadot2us-asm \
        vmadot3-lower vmadot3-translate vmadot3-asm vmadot3-run \
        vmadot3u-lower vmadot3u-translate vmadot3u-asm \
        vmadot3su-lower vmadot3su-translate vmadot3su-asm \
        vmadot3us-lower vmadot3us-translate vmadot3us-asm \
        linalg-matmul-lower linalg-matmul-translate linalg-matmul-asm \
        linalg-matmul-boundary-lower linalg-matmul-boundary-translate \
        linalg-matmul-boundary-asm linalg-matmul-boundary-run

all: vmadot vmadotu vmadotsu vmadotus

vmadot: vmadot.mlir
	$(BUDDY_OPT) $< -o vmadot-out.mlir

vmadotu: vmadotu.mlir
	$(BUDDY_OPT) $< -o vmadotu-out.mlir

vmadotsu: vmadotsu.mlir
	$(BUDDY_OPT) $< -o vmadotsu-out.mlir

vmadotus: vmadotus.mlir
	$(BUDDY_OPT) $< -o vmadotus-out.mlir

clean:
	rm -f *-out.mlir

vmadot-lower:
	@${BUDDY_OPT} ./vmadot.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadot-translate:
	@${BUDDY_OPT} ./vmadot.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadot-asm:
	@${BUDDY_OPT} ./vmadot.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

vmadot-run:
	@${BUDDY_OPT} ./vmadot_print_test.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o vmadot.o
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadot.o runtime_vmadot.c -o vmadot_test

vmadotu-lower:
	@${BUDDY_OPT} ./vmadotu.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadotu-translate:
	@${BUDDY_OPT} ./vmadotu.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadotu-asm:
	@${BUDDY_OPT} ./vmadotu.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

vmadotu-run:
	@${BUDDY_OPT} ./vmadotu_print_test.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o vmadotu.o
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadotu.o runtime_vmadotu.c -o vmadotu_test

vmadotsu-lower:
	@${BUDDY_OPT} ./vmadotsu.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadotsu-translate:
	@${BUDDY_OPT} ./vmadotsu.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadotsu-asm:
	@${BUDDY_OPT} ./vmadotsu.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

vmadotsu-run:
	@${BUDDY_OPT} ./vmadotsu_print_test.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o vmadotsu.o
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadotsu.o runtime_vmadotsu.c -o vmadotsu_test

vmadotus-lower:
	@${BUDDY_OPT} ./vmadotus.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadotus-translate:
	@${BUDDY_OPT} ./vmadotus.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadotus-asm:
	@${BUDDY_OPT} ./vmadotus.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

vmadotus-run:
	@${BUDDY_OPT} ./vmadotus_print_test.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o vmadotus.o
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadotus.o runtime_vmadotus.c -o vmadotus_test

vmadot1-lower:
	@${BUDDY_OPT} ./vmadot1.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadot1-translate:
	@${BUDDY_OPT} ./vmadot1.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadot1-asm:
	@${BUDDY_OPT} ./vmadot1.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

vmadot2-lower:
	@${BUDDY_OPT} ./vmadot2.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadot2-translate:
	@${BUDDY_OPT} ./vmadot2.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadot2-asm:
	@${BUDDY_OPT} ./vmadot2.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

vmadot3-lower:
	@${BUDDY_OPT} ./vmadot3.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadot3-translate:
	@${BUDDY_OPT} ./vmadot3.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadot3-asm:
	@${BUDDY_OPT} ./vmadot3.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

vmadot1-run:
	@${BUDDY_OPT} ./vmadot1_print_test.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o vmadot1.o
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadot1.o runtime_vmadot1.c -o vmadot1_test

vmadot2-run:
	@${BUDDY_OPT} ./vmadot2_print_test.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o vmadot2.o
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadot2.o runtime_vmadot2.c -o vmadot2_test

vmadot3-run:
	@${BUDDY_OPT} ./vmadot3_print_test.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o vmadot3.o
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static vmadot3.o runtime_vmadot3.c -o vmadot3_test

vmadot1u-lower:
	@${BUDDY_OPT} ./vmadot1u.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadot1u-translate:
	@${BUDDY_OPT} ./vmadot1u.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadot1u-asm:
	@${BUDDY_OPT} ./vmadot1u.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

vmadot1su-lower:
	@${BUDDY_OPT} ./vmadot1su.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadot1su-translate:
	@${BUDDY_OPT} ./vmadot1su.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadot1su-asm:
	@${BUDDY_OPT} ./vmadot1su.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

vmadot1us-lower:
	@${BUDDY_OPT} ./vmadot1us.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadot1us-translate:
	@${BUDDY_OPT} ./vmadot1us.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadot1us-asm:
	@${BUDDY_OPT} ./vmadot1us.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

vmadot2u-lower:
	@${BUDDY_OPT} ./vmadot2u.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadot2u-translate:
	@${BUDDY_OPT} ./vmadot2u.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadot2u-asm:
	@${BUDDY_OPT} ./vmadot2u.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

vmadot2su-lower:
	@${BUDDY_OPT} ./vmadot2su.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadot2su-translate:
	@${BUDDY_OPT} ./vmadot2su.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadot2su-asm:
	@${BUDDY_OPT} ./vmadot2su.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

vmadot2us-lower:
	@${BUDDY_OPT} ./vmadot2us.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadot2us-translate:
	@${BUDDY_OPT} ./vmadot2us.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadot2us-asm:
	@${BUDDY_OPT} ./vmadot2us.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

vmadot3u-lower:
	@${BUDDY_OPT} ./vmadot3u.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadot3u-translate:
	@${BUDDY_OPT} ./vmadot3u.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadot3u-asm:
	@${BUDDY_OPT} ./vmadot3u.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

vmadot3su-lower:
	@${BUDDY_OPT} ./vmadot3su.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadot3su-translate:
	@${BUDDY_OPT} ./vmadot3su.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadot3su-asm:
	@${BUDDY_OPT} ./vmadot3su.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

vmadot3us-lower:
	@${BUDDY_OPT} ./vmadot3us.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts \
		-o log.mlir

vmadot3us-translate:
	@${BUDDY_OPT} ./vmadot3us.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

vmadot3us-asm:
	@${BUDDY_OPT} ./vmadot3us.mlir \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

linalg-matmul-lower:
	@${BUDDY_OPT} ./linalg-to-ime-matmul.mlir \
		-lower-linalg-to-ime \
		-o log.mlir

linalg-matmul-translate:
	@${BUDDY_OPT} ./linalg-to-ime-matmul.mlir \
		-lower-linalg-to-ime \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-matmul-asm:
	@${BUDDY_OPT} ./linalg-to-ime-matmul.mlir \
		-lower-linalg-to-ime \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

linalg-matmul-boundary-lower:
	@${BUDDY_OPT} ./linalg-to-ime-matmul-boundary.mlir \
		-lower-linalg-to-ime \
		-o log.mlir

linalg-matmul-boundary-translate:
	@${BUDDY_OPT} ./linalg-to-ime-matmul-boundary.mlir \
		-lower-linalg-to-ime \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-expand-strided-metadata \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

linalg-matmul-boundary-asm:
	@${BUDDY_OPT} ./linalg-to-ime-matmul-boundary.mlir \
		-lower-linalg-to-ime \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-expand-strided-metadata \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o log.s

linalg-matmul-boundary-run:
	@${BUDDY_OPT} ./linalg-to-ime-matmul-boundary-func.mlir \
		-lower-linalg-to-ime \
		-lower-ime \
		-convert-linalg-to-loops \
		-lower-affine \
		-expand-strided-metadata \
		-convert-scf-to-cf \
		-convert-cf-to-llvm \
		-convert-arith-to-llvm \
		-convert-math-to-llvm \
		-convert-func-to-llvm \
		-finalize-memref-to-llvm \
		-reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+m,+v,+xsmtvdot \
		-o matmul_boundary.o
	riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static \
		matmul_boundary.s runtime_matmul_boundary.c \
		-o matmul_boundary_test

# ===========================================================================
# FP16 matmul targets (require -mattr=+zvfh for half-precision float support)
# ===========================================================================


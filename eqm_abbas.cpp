#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/utility.hpp>
#include"custom-modules/eqm_abbas.hpp"



namespace custom_models{

	class eqm_fmap : public std::vector<QuantumCircuit*>
	{
		public:
			/*
			 *@param batched input tensor of sizes (batch,sin)
			 *
			 *
			 */
			eqm_fmap(const at::Tensor &x){
				PRINTT(x,"Tensor(BATCH,SIN)");
				for(auto i=0;i<x.size(0);i++)
				{
					push_back(new QuantumCircuit(x.size(1)));
				}
				const auto unbinded=torch::unbind(x);
				std::transform(EXEC,unbinded.begin(),unbinded.end(),begin(),begin(),[](auto x,auto circuit)
						{

						torch::Tensor ten =x.to(torch::TensorOptions(torch::kCPU).dtype(at::kDouble));
						auto at = ten.accessor<double,1>();

						for(int i=0;i<x.size(0);++i)
						{
						circuit->add_H_gate(i);
						circuit->add_RZ_gate(i,-M_PI*at[i]);
						}
						return circuit;
						});

			};
			~eqm_fmap(void){
				std::for_each(begin(),end(),[](auto item){delete item;});
			}
	};
	class eqm_Vcircuit : public QuantumCircuit
	{
		public:
			eqm_Vcircuit(const torch::Tensor& w, const int64_t sin):QuantumCircuit(sin){
				PRINTT(w,"Tensor(D_PHI)");
				size_t ind=0;
				torch::Tensor ten =w.to(torch::TensorOptions(torch::kCPU).dtype(at::kDouble));
				auto at = ten.accessor<double,1>();


				for(int i=0;i<(w.size(0)/sin-1);i++){
					for(int j=0;j<sin;j++){
						add_RY_gate(j,-M_PI*at[ind]);
						ind++;
					}
					for(int j=0;j<sin;j++){
						for(int k=j+1;k<sin;k++){
							add_CNOT_gate(j,k);
						}
					}
				}
				for(int j=0;j<sin;j++){
					add_RY_gate(j,-M_PI*at[ind]);
					ind++;
				}

			};

	};

	class black_box : public torch::autograd::Function<black_box> {
		public:
			static torch::Tensor forward(torch::autograd::AutogradContext *ctx, const torch::Tensor& W,const at::Tensor & x);

			static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list grad_outputs) ;

			static torch::Tensor run(const at::Tensor &x, const torch::Tensor W);
	};
	EQM_abbasImpl::EQM_abbasImpl(int64_t Sin, int64_t Sout, int64_t D_phi):sin(Sin),sout(Sout),d_phi(D_phi),
	weights(register_parameter("weights",2*torch::rand(d_phi)-1))
	{

	};
	torch::Tensor black_box::forward(torch::autograd::AutogradContext *ctx, const torch::Tensor& W,const at::Tensor & x)
	{
		ctx->save_for_backward({W});
		ctx->saved_data["x"] = x;
		const auto var=run(x,W);
		PRINTT(var,"Tensor(BATCH,SOUT)");
		return var;//run(x,W);

	}
	size_t upbits(size_t i)
	{
		size_t count=0;
		while(i)
		{
			count+=i&1;
			i>>=1;
		}
		return count;
	}

	torch::Tensor black_box::run(const at::Tensor &x, const torch::Tensor W)
	{
		torch::NoGradGuard no_grad;
		auto feu=eqm_fmap(x);
		std::vector<torch::Tensor> result_vector(feu.size());
		auto wei=eqm_Vcircuit(W,x.size(1));
		std::transform(EXEC,feu.begin(),feu.end(),result_vector.begin(),[&](const auto &feu){
				QuantumState state(x.size(1));
				state.set_zero_state();

				feu->update_quantum_state(&state);
				wei.update_quantum_state(&state);

				std::vector<std::complex<double>> elem(1UL<<x.size(1));
				std::vector<size_t> index(1UL<<x.size(1));
				std::iota(index.begin(),index.end(),0);
				const CPPCTYPE* raw_data_cpp = state.data_cpp();
				std::move(EXEC,raw_data_cpp,raw_data_cpp+(1UL<<x.size(1)),elem.begin());

				auto prob=std::transform_reduce(EXEC,elem.begin(),elem.end(),index.begin(),0.0, std::plus<double>(),
						[](const auto &item,const auto &index){return (upbits(index)%2)?std::norm(item):0.0;});

				auto var=at::zeros({2},torch::requires_grad());
				var[0]=prob;
				var[1]=1-prob;
				PRINTT(var,"Tensor(SOUT)");
				return var;

		});
		return torch::stack(result_vector);
	}
	torch::autograd::tensor_list black_box::backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list grad_outputs) {
		torch::NoGradGuard no_grad;
		const auto saved = ctx->get_saved_variables();
		const auto W = saved[0];
		const auto x=(ctx->saved_data["x"].toTensor());

		std::vector<at::Tensor> grads(W.size(0));

		std::vector<double> index(W.size(0));
		std::iota(index.begin(), index.end(), 0);
		std::transform(EXEC,index.begin(),index.end(),grads.begin(),[&x,W]
				(const auto &ind)
				{
				torch::NoGradGuard no_grad;
				auto W1=W.to(torch::TensorOptions(torch::kCPU).dtype(at::kDouble));
				auto at = W1.accessor<double,1>();
				at[ind]+=M_PI/2.0;
				const auto right=run(x,W1);
				at[ind]-=M_PI;
				const auto left=run(x,W1);
				return (right-left)*M_PI/2.0;
				});

		auto result=torch::stack(grads);
		PRINTT(result,"Tensor(DPHI_,BATCH,SOUT)");
		PRINTT(grad_outputs[0],"Tensor(BATCH,SOUT)");
		auto der=torch::einsum("abc,bc->a",{result,grad_outputs[0]});
		PRINTT(der[0],"Tensor(DPHI_)");
		return {der,torch::Tensor()}; //check this
	}

	torch::Tensor EQM_abbasImpl::forward(const at::Tensor & x)
	{
		return black_box::apply(weights,x);
	}
};

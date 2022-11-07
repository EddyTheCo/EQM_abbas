
#pragma once

#include <torch/torch.h>

#ifdef USE_YAML
#include<yaml-cpp/yaml.h>
#endif


namespace custom_models{

    class EQM_abbasImpl : public torch::nn::Module, torch::autograd::Function<EQM_abbasImpl>   {
		public:
			/// Create a Quantum neural network model following https://doi.org/10.48550/arXiv.2011.00027
			///
			EQM_abbasImpl(int64_t Sin, int64_t Sout, int64_t D_phi);

#ifdef USE_YAML
            EQM_abbasImpl(YAML::Node& config):EQM_abbasImpl((config["Sin"]).as<int64_t>(),
					(config["Sout"]).as<int64_t>(),
					(config["D_phi"]).as<int64_t>(),
					){};
#endif
			torch::Tensor forward(torch::autograd::AutogradContext *ctx,const at::Tensor & x);
			torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list grad_outputs) ;
            torch::Tensor run(const at::Tensor &x, const torch::Tensor W);
		private:
			const int64_t  sin,sout,d_phi;
			torch::Tensor weights;
	};
	TORCH_MODULE(EQM_abbas);
};



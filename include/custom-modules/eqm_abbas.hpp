
#pragma once

#include <torch/torch.h>

#ifdef USE_YAML
#include<yaml-cpp/yaml.h>
#endif

#ifdef _DEBUG
#define EXEC oneapi::dpl::execution::seq
#define PRINTT(x,str) std::cout<<#x<<x.sizes()<<x.dtype()<<" = "<<str<<std::endl; \
	//
#else
#define EXEC oneapi::dpl::execution::par_unseq
#define PRINTT(x,str) //
#endif

namespace custom_models{

    class EQM_abbasImpl : public torch::nn::Module  {
		public:
			/// Create a Quantum neural network model following https://doi.org/10.48550/arXiv.2011.00027
			///
			EQM_abbasImpl(int64_t Sin, int64_t Sout, int64_t D_phi);

#ifdef USE_YAML
            EQM_abbasImpl(YAML::Node config):EQM_abbasImpl((config["Sin"]).as<int64_t>(),
					(config["Sout"]).as<int64_t>(),
					(config["D_phi"]).as<int64_t>()
					){};
#endif
			torch::Tensor forward(const at::Tensor & x);
		private:
			const int64_t  sin,sout,d_phi;
			torch::Tensor weights;
	};
	TORCH_MODULE(EQM_abbas);
};



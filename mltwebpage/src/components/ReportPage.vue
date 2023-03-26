<template>
  <div class="home">
    <div class="common-layout">

      <el-container>
      <el-container>
        <el-header>
          <h1 id="Report For CSE 274">Implementation of BDPT and MMLT on GPU with<br/> Simple Primal Sample Space Latent Nueral Mutation</h1>
        </el-header>
        <el-container>
          <el-container>
            <el-main>
              <el-row class="row-bg"><br/></el-row>
              <!-- Chapter 1: Introduction -->
              <el-row class="row-bg">
                <h2 id="1. Introduction" style="text-align:left">1. Introduction</h2>
                <p style="text-align:left">
                  In this project, I implemented BDPT (Bidirectional Path Tracing) <a href="#cite-veach">[1]</a> 
                  and MMLT (Multiplexed metropolis Light Transport) <a href="#cite-mmlt">[2]</a> on my GPU renderer,
                  <a href="https://github.com/SuikaSibyl/SIByLEngine2023">SIByL Engine</a>. 
                  Also I have some naive ideas and experiments on Neural Mutation for Metropolis light transport
                  as well as common Metropolis-Hastings sampling.
                </p>
                <p style="text-align:left">
                  The render part is implemented in C++ with Vulkan, and the implementation is only compatible with Windows OS and Nvidia RTX GPU.
                  All the results are run and measured on my personal RTX 3070 laptop. The neural part is implemented in Python with PyTorch.
                  The inter-process communication is done with localhost socket.
                </p>
                <p style="text-align:left">
                  In <a href="#2.">Section 2</a> I will breifly talk about how I implement BDPT and MMLT on GPU, without lots of details, and show
                  some results and comparison with unidirectional path tracing.
                </p>
                <p style="text-align:left">
                  Then, <a href="#3.">Section 3</a> discussed something about normalizing flow and neural importance sampling. I will also show some results
                  and comparison in 2D monochrome image case. They are talked about because they are the basis of my "Latent Space Mutation" idea.
                </p>
                <p style="text-align:left">
                  And <a href="#4.">Section 4</a> talks about "Latent Space Mutation", a (probably) new concept I proposed, and why I think it might be useful.
                  Some experiments are also shown both in 2D monochrome image case and single-depth MMLT case.
                </p>
              </el-row>
              <!-- Chapter 2: BDPT & MMLT -->
              <el-row class="row-bg" justify="left">
                <h2 id="2.">2. BDPT & MMLT on GPU</h2>
              </el-row>
              <el-row class="row-bg" justify="left">
                <h3 id="2.1" style="margin-top: 0in;">2.1 Brief Introduction</h3>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  BDPT and MLT are proposed by Veach <a href="#cite-veach">[1]</a>. PSSMLT <a href="#cite-pssmlt">[4]</a> is a simple version of the vanilla
                  MLT that do mutation in primal sample space. And MMLT <a href="#cite-mmlt">[2]</a> is a different way of doing PSSMLT, that has a static
                  path depth for each Markov Chain. I implement both BDPT and MLT in my renderer referring to the implementation of PBRT <a href="#cite-pbrt">[3]</a>.
                  Volumetric rendering is not supported in this project.
                </p>
              </el-row>

              <el-row class="row-bg" justify="left">
                <h3 id="2.1" style="margin-top: 0in;">2.2 Implementation and Experiments</h3>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  For both BDPT and MLT, splatting is required for contributing to the film. As far as I could tell, there are three ways to do splatting.
                  (1) we could use atomicAdd to add the contribution to the film, (2) we could use a atomic buffer to do per-pixel spinlock and do mutex adding,
                  (3) we could use a temporary buffer to store the contribution, and then
                  dispatch a set of pixel-size fragment shader to do adding alpah blending. I use the first method, although it need three atomic buffer instead 
                  of one, the performance is probabily better than the second method, I actually could not compare it because my per-pixel spinlock get deadlock
                  on raygen shader warps although it works fine for compute shader...
                </p>
                <p style="text-align:left">
                  To implement BDPT, I use only one large kernel, tracing 1 spp BDPT for each pixel. I did not optimize the kernel too much, but reduce the
                  original cost of 170+ ms per frame to 10+ms per frame by a simple trick: manually unrolling the loop for path connection. The main reason 
                  is not carefully tested, but I guess it is because unrolling prevent some fake loop carried dependency, which is harmful for pipeline.
                  I just observed severe long scoreboard stall when I use the original loop, and the stall is gone after unrolling.
                </p>
                <p style="text-align:left">
                  To implement MMLT, I designed a dynamic pass pipeline for interactive frame rate. The first part should be boostrap sample generation.
                  We should generate more boostrap samples to choosen from and compute average b, but running all samples in one frame is too slow. Therefore
                  we could armotize the cost by running a few samples in each frame. After some frames, we then begin mutation pass. For each frame we run
                  one mutation for one Markov Chain per thread. Notice that the first frame in "mutation" pass is not mutation, but choose samples from boostrap.
                  I use hierarchical 2d mip for sampling boostrap samples. The per-frame task assignment is shown below:
                </p>
                <el-row class="row-bg" justify="center">
                <el-col :span="17">
                  <el-image style="align:center" src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/20230324094213.png" />
                  <p><code>
                    My cross-frame MMLT pipeline.
                  </code></p>
                  </el-col>
                </el-row>
                <p style="text-align:left">
                  An interesting design I want to mention is how to get PSS sample from boostrap. We have far more boostrap samples than chains per frame,
                  if we store the PSS samples for boostrap samples, it would need a temporary buffer which is large and would not be used later. Instead,
                  I recover the random seed for the boostrap sample chosen, and generate the PSS sample sequence on the fly. 
                </p>
                <p style="text-align:left">
                  BDPT has better convergence rate than UDPT in many cases. In the demo scene shown below, BDPT is also slightly better than UDPT. The scene is just
                  a simple set of still lifes in the room from Veach's MLT scene. The only light is a sphere in the next room, and the camera is in the room with the still lifes.
                  The scene is rendered with 500 spp, and the result is shown below.
                </p>
              </el-row>
              <el-row class="row-bg" justify="center">
                <el-col>
                  <div id="udptvsbdpt"></div>
                  <p><code>left: unidirectional 500 spp | right: bidirectional 500 spp</code></p>
                </el-col>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  MMLT has better results than BDPT, as shown in the figure below. (Even though lookes like to have more fireflies somehow.)
                  We could clearly see that MLT is less noisy than BDPT under same spp. But as I am using MMLT, it actually take much more time because
                  it need at least 4 times path numbers as MMLT trace one path for each depth, so for a 3-bounce path, it need 3 more paths per sample.
                  And I realize that this dummy MMLT is less efficient thant BDPT, as BDPT actually has some kind of coherency in neighbor paths,
                  this dummy MMLT is completely random and has larger divergence.
                </p>
              </el-row>
              <el-row class="row-bg" justify="center">
                <el-col>
                  <div id="bdptvsmlt"></div>
                  <p><code>left: bidirectional 500 spp | right: mmlt 500 mpp</code></p>
                </el-col>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  Actually I should do some obvious optimizing for MMLT on GPU. As for each Markov Chain, the depth is fixed, so we could actually do some
                  sort after choosing boostrap path, and do 4 different tracing drawcall to issue 4 bundles of chains with different depth. This could reduce lots of unnecessary
                  thread divergence, and easily achieve coherent with only one sort (actually only one radix sort pass or some compaction passes) per rendering, which is much simpler than optimizing BDPT for GPU.
                </p>
                <p style="text-align:left">
                  In my observation, thread divergence is the main problem for BDPT and MLT, so I beleive this optimization is really worth to implement 
                  and could hopefully boost the performance. But as the deadline is approaching, I have to leave this optimization for future work.
                </p>
              </el-row>

                <el-row class="row-bg" justify="center">
                  <el-col :span="17">
                    <iframe width="560" height="315" src="https://www.youtube.com/embed/tmI0e2OMLN4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
                  </el-col>
                </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  <br/>Both algorithms are integrated into my interactive renderer, as shown in the video. I guess the one of the most important target for porting these algorithm into
                  an interactive renderer is to limit the cost per frame. Otherwise a long stall would get the whole system stuck. In cpu applications we could use multi-threading to
                  decouple UI and rendering, but as I am using ImGui, getting the GPU queue stack would also affect the UI. And of course it would be important to optimize the code
                  for GPU architecture for better performance.
                </p>
              </el-row>

              <!-- Chapter 3: Neural Importance Sampling -->
              <el-row class="row-bg" justify="left">
                <h2 id="3.">3. Neural Importance Sampling</h2>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  The object of the project is about exploring the neural mutation for MLT, so it might seems a little bit off-topic to talk about neural importance sampling here.
                  But as normalizing flow is a very useful technique for generative models, I think it is worth to talk about it a little bit.
                  Neural importance sampling is probably the first work that I know that uses normalizing flow in rendering,
                  and I personally think it could be used in neural mutation, see <a href="#4.">Section 4</a> for more details.
                </p>
              </el-row>
              <el-row class="row-bg" justify="left">
                <h3 id="3.1">3.1 Theory behind NIS</h3>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  Normalizing flow is an useful technique to generate samples subject to some certain distributions by a neural network.
                  But the really interesting part that makes it different from other generative models is that it actually creates a
                  bijection between two distributions, which means the jacobian of the transformation is computable.
                </p>
                <p style="text-align:left">
                  NICE <a href="#cite-nice">[5]</a> proposed a simple way to construct a bijection by using a structure called "Coupling Layer".
                  RealNVP <a href="#cite-realNVP">[6]</a> further extend that into "Affine Coupling Layer", which introduces non-volume-preserving transformation in each layer and is thus more powerful.
                  GLOW <a href="#cite-glow">[7]</a> proposed another primitive to construct a bjicetion, the invertible 1x1 Convolutions.
                </p>
                <p style="text-align:left">
                  Müller et al. <a href="#cite-nis">[8]</a> and Zheng et al. <a href="#cite-realNVP">[9]</a> introduce normalizing flow into rendering and use it for neural importance sampling.
                  Generally speaking, both of them use a normalizing flow to build a bijection between [0,1] uniform distribution and the target radiance distribution, so that they could do importance sampling.
                </p>
                <p style="text-align:left">
                  Zheng et al. <a href="#cite-realNVP">[9]</a> did a simpler but inspiring work. They basically use the RealNVP to importance sample the primal space vector for path construction.
                  It really provides a lot technical details for me to implement this stuff, and actually what I implemented in PyTorch is very similar to their work.
                  A practical contribution is about how to take normalizing flow which is primarily on <math-jax latex="{\mathbb{R}}^n"/> into <math-jax latex="[0,1]^n" />.
                </p>
              </el-row>

              <el-row class="row-bg" justify="center">
                <el-image style="align:center" src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/20230321230003.png" />
                <p><code>
                  Computational structure of Zheng et al.'s neural importance sampling model based on Real NVP <br/> 
                  from <a href="#cite-realNVP">[9]</a> .
                  I implement almost the same thing in PyTorch for later experiments.
                </code></p>
              </el-row>
              
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  They use a scaling and logit and their inverse in the start and their inverse (sigmoid and inverse scaling) in the end of the flow to mapping between 
                  <math-jax latex="[0,1]^n" /> and <math-jax latex="{\mathbb{R}}^n"/>. The scaling layer is actually a constant scaling tha maps <math-jax latex="[0,1]"/> 
                  to <math-jax latex="[\epsilon,1-\epsilon]"/> which prevents logit producing too large number, which will cause severe numerical problem and introducing
                  more NaN in both results and gradients.
                </p>
                <p style="text-align:left">
                  However, in training, they use the traditional loss function for normalizing flow, a.k.a. maximizing the log likelihood of generating target distribution.
                  This implies we need to sample from the target distribution, which is not straightforward in rendering. But we could still do it by some indirect way, like
                  MCMC or resampling <a href="#cite-ris">[10]</a>.
                </p>
                <p style="text-align:left">
                  Müller et al. <a href="#cite-nis">[8]</a> gives some advanced contributions including a more powerful piecewise-polynomial coupling layer,
                  and deriving two new loss functions for neural importance sampling. Interestingly, they showed that minimizing the KL divergence amounts to maximizing
                  a weighted log likelihood.
                </p>
                <p style="text-align:left">
                  The KL divergence between target distribution <math-jax latex="p(x)"/> and the learned distribution <math-jax latex="q(x)"/> is:
                </p>
              </el-row>
              <el-row class="row-bg" justify="center">
                <p>
                  <math-jax latex="$$
                  \begin{aligned}
                  KL(p||q;\theta) &= \int_\Omega p(x) \log \frac{p(x)}{q(x;\theta)} {\rm d}x\\
                                  &= \int_\Omega p(x) \log p(x) {\rm d}x - \int_\Omega p(x) \log q(x;\theta) {\rm d}x\\
                  \end{aligned}
                  $$" />
                </p>
              </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  In practice we could not get <math-jax latex="p(x)"/> as we do not know the normalizing term, and we could not differentiate the <math-jax latex="f(x)"/>
                  with respect to <math-jax latex="x"/> (<del>although I guess using differentiable rendering for PSS is probabaly helpful?</del>). Anyway only the second
                  term, the cross entropy term, has gradient. And the exciting part is:
                </p>
              </el-row>

              <el-row class="row-bg" justify="center">
                <p>
                  <math-jax latex="$$
                  \begin{aligned}
                  \nabla_\theta KL(p||q;\theta) &= -\nabla_\theta \int_\Omega p(x)\log q(x;\theta) {\rm d}x\\
                                  &= \mathbb{E}_{q(x;\theta)} \left[-\frac{p(X)}{q(X;\theta)} \nabla_\theta \log q(X;\theta) \right]\\
                  \end{aligned}
                  $$" />
                </p>
              </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  where the expectation is over <math-jax latex="X\sim q(x;\theta) "/>, i.e. the samples are drawn from the learned generative model.
                  Even if we could not know the normalizing term <math-jax latex="F"/>, we could still simply use  <math-jax latex="f(x)=F\cdot p(x)"/> to substitute 
                  <math-jax latex="p(x)"/>, as it would only introduce a constant factor in the gradient, which is not harmful for gradient descent.
                  It just shows that minimizing the KL divergence via gradient descent is equivalent to minimizing the negative log likelihood weighted by Monte Carlo estimates of F .
                </p>
              </el-row>

              <el-row class="row-bg" justify="left">
                <h3 id="3.2">3.2 Implementation and Experiments</h3>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  I implemented NICE <a href="#cite-nice">[5]</a> and RealNVP <a href="#cite-realNVP">[6]</a> in PyTorch framework.
                  The specific implementation details for t/s transform are not declared in the papers, for NICE I use MLP for t transform, and for RealNVP I use 
                  a 2-MLP residual blocks for t-s transform (as Zheng et al. <a href="#cite-realNVP">[9]</a> did). As RealNVP is more powerful than NICE,
                  all the following experiments are based on RealNVP.
                </p>
                <p style="text-align:left">
                  First, I checked how the number of layers influence the expressivity. The "converged" results are shown in the following figure.
                  But notice that all the results are not fully converged, I just cut the training process after their convergence becomes super slow.
                </p>
              </el-row>
              <el-row class="row-bg" justify="center">
                <el-image style="align:center" src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/expressivity.png" />
                <p><code>
                  (0): the target image  <a href="https://pixabay.com/photos/statue-sculpture-figure-1275469/">stone sculpture</a> / distribution (rings) <br/> 
                  (1): the same targets, but here darker pixels refers to higher density. <br/>
                  (2): (quasi) convergenced distribution by 4 coupling layer RealNVP. <br/>
                  (3): (quasi) convergenced distribution by 8 coupling layer RealNVP.
                </code></p>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  We could clearly find that more numbers of coupling layers could have stronger expresivity. (a/b-3) fits the objective distribution better thant (a/b-2).
                  This is exactly what we expected. It is said that Piecewise-polynomial coupling layer by Müller et al. <a href="#cite-nis">[8]</a> has better expressivity
                  for each layer, and thus could reduce the total number of layers, but I did not implement it here.
                </p>
                <p style="text-align:left">
                  Before training towards any complex distribution, we first start from a checkpoint that fits uniform bijection well.
                  The process of training towards a uniform distribution is shown below:
                </p>
              </el-row>
              <el-row class="row-bg" justify="center">
                <el-image style="align:center" src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/uniform%2Btrain.gif" />
                <p><code>
                  How RealNVP progressively learned to be a uniform bijection.
                </code></p>
              </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  Then we train towards the target distribution. There are two ways to do it: <br/>
                  (1) <b>Inverse Training</b>: Traditional way to do it in normalizing flow. Given X samples subject to target distribution, use Inverse
                  transform to find corresponding latent vairable Z, and maximize the likelihood of generating X. Here we generate X samples by Metropolis-Hastings sampling
                  with average 20 mutations per pixel. <br/>
                  (2) <b>Forward Training:</b>: According to Müller et al. <a href="#cite-nis">[8]</a> it is also possible to start from Z samples subject to priori distribution,
                  use forward transform to find corresponding X samples, and maximize the KL divergence between F(X) values and pdf of X. In this process we do not need to generate
                  samples subject to target distribution.<br/>
                </p>
                <p style="text-align:left">
                  I tried both and the results are shown below, both of them are using 16 affine coupling layers. I am <b>NOT</b> sure that there is no bug in forward training.
                  But it seems that even though forward training could capture the main structure of the target distribution, it could hardly know where to go and get stuck
                  at some low frequency status. This is not expected, because in Müller et al. <a href="#cite-nis">[8]</a> get fairly not-that-bad results even when using affine layers.
                  For further investigation, I will try to implement Piecewise-polynomial coupling layer and check the correctness of forward training code.
                  But for the following experiments, I will only use inverse training for quality and simplicity.
                </p>
              </el-row>  
              <el-row class="row-bg" justify="center">
                <el-image style="align:center" src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/image_train.gif" />
                <p><code>
                  How RealNVP progressively learned to be a uniform bijection.
                </code></p>
              </el-row>

              <!-- Chapter 4: MCMC Neural Mutation -->
              <el-row class="row-bg" justify="left">
                <h2 id="4.">4. Latent Space Mutation</h2>
              </el-row>
              <el-row class="row-bg" justify="left">
                <h3 id="4.1" style="margin-top: 0in;">4.1 What Is & Why Latent Space Mutation</h3>
              </el-row>
              <el-row class="row-bg" justify="left">
                <h4 style="margin-top: 0.1in;">What is Latent Space Mutation<br/></h4>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  I proposed this concept to hopefully improve mutation for MLT, which is some kind of combination of importance sampling and Metropolis-Hastings sampling.
                  I think the two closely related concepts are neural importance sampling and hidden markov model.
                </p>
                </el-row>
              <el-row class="row-bg" justify="center">
                <el-col :span="8">
                  <el-image style="align:center" src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/20230325132820.png" />
                  <p><code>
                    Vanilla Metropolis-Hastings.
                  </code></p>
                  </el-col>
                </el-row>
              <el-row class="row-bg" justify="center">
                <el-col :span="8">
                  <el-image style="align:center" src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/20230325132813.png" />
                  <p><code>
                    Hidden Markov Model.
                  </code></p>
                  </el-col>
                </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  The original way of doing Metropolis-Hastings is tracking a Markov Chain on primal state x. But in Hidden Markov Model,
                  we track a hidden state instead and generate X from another distribution <math-jax latex="p(x|z)"/>. Now let's substitute
                  it with a known bijection defined by the neural network for neural importance sampling.
                </p>
              </el-row>
              <el-row class="row-bg" justify="center">
                <el-col :span="8">
                  <el-image style="align:center" src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/20230325133905.png" />
                  <p><code>
                    Vanilla Metropolis-Hastings.
                  </code></p>
                  </el-col>
                </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  Ideally the neural network should provides a bijection maps a uniform distribution Z to target distribution X, by some kind of warping.
                  Thus we could do mutation like this:<br/>
                  (1) Start from the start state <math-jax latex="X"/>. <br/>
                  (2) Mapping to latent space <math-jax latex="Z=g(X)"/> . <br/>
                  (3) Do common mutation in latent space <math-jax latex="Z'=mutate(Z)"/>. <br/>
                  (4) Mapping to primal space <math-jax latex="X'=f(Z')"/> . <br/>
                </p>
                <p style="text-align:left">
                  Therefore we essentially want to do mutation in latent space instead of primal space. Here, the "common strategy" in step (3) refers to simple strategies like
                  a uniform sampling large step and a normal sampling small step, mentioned in PBRT <a href="#cite-pbrt">[3]</a>.
                </p>
                <p style="text-align:left">
                  But why should we do this? We could see how large step and small step works in latent space to get intuition.
                </p>
              </el-row>

              <el-row class="row-bg" justify="left">
                <h4 style="margin-top: 0.in;">Ideal Large Step Mutation</h4>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  Let's start with something simpler, so consider the large step mutation first. In primal space, we uniform sample a random vector as proposal, which is very likely to be rejected.
                  Using latent space mutation, we first generate a random vector in latent space and then map into primal space. Ideally, this mapping <math-jax latex="g(z)"/> could produce
                  x porportial to target distribution <math-jax latex="p"/>.
                </p>
                <p style="text-align:left">
                  Considering detailed balance condition for <math-jax latex="X"/> could be a little bit confusing, as describing <math-jax latex="T(X\rightarrow X')"/> is not clear.
                  But we could consider the latent/hidden Markov chain, the underlying latent distribution <math-jax latex="Z"/> is also equilibrium to a uniform distribution.
                  <math-jax latex="f(Z)=f(Z')=1"/> and <math-jax latex="T(Z\rightarrow Z')=T(Z'\rightarrow Z)=1"/>, thus the classical acceptance ratio is:
                  <math-jax latex="a(Z\rightarrow Z')=1"/>.
                </p>
                <p style="text-align:left">
                  Thus what actually happen is we proporse a random vector proportional to <math-jax latex="p"/>, and then we always accept it. This is exactly
                  the process of optimal importance sampling, which is the best way to do sampling. It does immediate convergence and has no correlation between samples.
                </p>
              </el-row>

              <el-row class="row-bg" justify="left">
                <h4 style="margin-top: 0.in;">Ideal Small Step Mutation</h4>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  In small step case, things become a little bit subtle. Ideally, the mapping from Z to X is some kind of warping that makes 
                  uniform Z concentrated to the dense regions of X. In a ideal case the mapping is super smooth and regular (in practice, RealNVP could hardly achieve this).
                </p>
              </el-row>
              <el-row class="row-bg" justify="center">
                <el-col :span="14">
                  <el-image style="align:center" src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/20230325135516.png" />
                  <p><code>
                    Mapping from Z to X is some warping.
                  </code></p>
                  </el-col>
                </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  But let' assume this happens. We do a isotropic Gaussian from the start state (red point) and produce two possible mutated positions (blue and green points).
                  If we do this mutation is primal step, they would located in low-probability regions and are likely to be rejected. On the other hand, if we do this mutation in latent space,
                  This samples would be warpped by the mapping and agian located in high-probability regions. The rotation and scaling of the Gaussian are just adaptive,
                  a narrow axis would be squeezed by the warping and result in a more narrow Gaussian, as shown in the figure below.
                </p>
              </el-row>
              <el-row class="row-bg" justify="center">
                <el-col :span="14">
                  <el-image style="align:center" src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/20230325135315.png" />
                  <p><code>
                    The first row: latent space mutation. <br/>
                    The second row: primal space mutation.
                  </code></p>
                  </el-col>
                </el-row>
                <p style="text-align:left">
                  Again when the underlying latent distribution <math-jax latex="Z"/> is equilibrium to a uniform distribution,
                  <math-jax latex="f(Z)=f(Z')=1"/> and <math-jax latex="T(Z\rightarrow Z')=T(Z'\rightarrow Z)=1"/>, and the classical acceptance ratio is again:
                  <math-jax latex="a(Z\rightarrow Z')=1"/>. Which means we has 1 acceptance rate and 0 rejection rate, while 
                  still be able to exporling the neighbor space.
                </p>

                <el-row class="row-bg" justify="left">
                  <h4 style="margin-top: 0.in;">To be NOT ideal</h4>
                </el-row>
                <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  Interestingly, we could see that in ideal case, both large step and small step mutations are just importance sampling. The large step is 
                  just how we commonly do importance sampling, while the small step is a random walk version of it. The isotropic random walk in latent space finally
                  would create a sequence of uniform samples in latent space and would be mapped to exactly the target distribution in primal space.
                </p>
                <p style="text-align:left">
                  The truth is, things are NOT ideal. But this truth would not frustrate us, we could see that sub-optimal importance sampling still helps us a lot in
                  convergence rate. Thus we could expect that a suboptimal neural (and even non-neural) latent space mutation would also help us, as some kind of 
                  interpolation between optimal importance sampling and dummy metropolis-hastings.
                </p>
                <p style="text-align:left">
                  Something different is we could no longer do acceptance in latent space. As the mapping is not perfect, a uniform sampling in Z would not
                  recover the correct p(x) in primal space, so we should do acceptancein primal space. But this would introduce a new problem: we must
                  evaluate the transition probability <math-jax latex="T(X\rightarrow X')"/>. I am not sure in this part, it might be similar to primal mutation or need some additional
                  Jacobian, I need to do more experiments on this.
                </p>
              </el-row>

              <el-row class="row-bg" justify="left">
                <h3 id="4.2" style="margin-top: 0in;">4.2 Monochrome 2D Distribution</h3>
              </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  To quickly verify the design, we do experiments on a simple monochrome 2D distribution.
                </p>
                <p style="text-align:left">
                  The distribution tested is an anisotropy oval, the target distribution and the learned results are: 
                </p>
              </el-row>
              <el-row class="row-bg" justify="center">
                <el-col :span="10">
                  <el-image style="align:center" src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/20230325213419.png" />
                  <p><code>
                    Left: the target distribution <br/>
                    Right : the learned distribution.
                  </code></p>
                  </el-col>
                </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  The results are also very good. We could observe both reconstruction results and acceptance rates are relatively better when we adopt
                  the neural latent space sampling:
                </p>
              </el-row>
              <el-row class="row-bg" justify="center">
                <el-col :span="12">
                  <el-image style="align:center" src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/20230325213359.png" />
                  </el-col>
                </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                </p>
              </el-row>
              
              <el-row class="row-bg" justify="left">
                <h3 id="4.2" style="margin-top: 0in;">4.3 Neural Mutate the Metropolis Light Transport</h3>
              </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  I also tried to use the neural mutation to mutate the Metropolis Light Transport. The result is not very good, but I think it is still worth to share.
                </p>
                <p style="text-align:left">
                  It is actually quite a disaster when implementating this: NaNs are everywhere. As soon as I raise the dimension from 2D to 48-dim, the gradient becomes unstable,
                  if the layers are too deep, NaNs will quickly appear in gradients and then in loss, as far as I checked it is kind of related to logit-sigmoid mapping
                  which do exponential operations. As a result I only use 4 coupling layers with 48 dimensions (I use at most 48 PSS random variables for a path).
                </p>
                <p style="text-align:left">
                  For MMLT, we have different depth so I think we should use different models for different depths. We could consider a path that is important for depth 4,
                  then its subpath would not be important for depth 3 unless the last vertex is also on light source. In this experiment, I only test the case that has depth 4.
                </p>
                <p style="text-align:left">
                  The test scene is a glass Stanford bunny, we could see that most important path with depth 4 are on the bunny:
                </p>
              </el-row>
              
              <el-row class="row-bg" justify="center">
                <el-col>
                  <div id="alldepthvsdpeth4"></div>
                  <p><code>left: result with depth 0-4 | right: only using depth 4</code></p>
                </el-col>
              </el-row>


              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  Bothered by the NaNs, I only use 4 layers so the expressivity is significantly not that good in high dimensions. The figure below
                  shows the comparison between the target distribution and the learned distribution. The target distribution are generated by the
                  MLT, and all the images are showing their projection on 2 of 48 dims. 
                </p>
              </el-row>

              <el-row class="row-bg" justify="center">
                <el-col :span="11">
                  <el-image style="align:center" src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/learned.png" />
                  <p><code>
                    Comparison between target and learned distribution. Notice that the target distribution are curved to show subtle details.
                  </code></p>
                </el-col>
              </el-row>
              
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  And in actual rendering we need to input current sample stream to the neural network every frame, so we need to do 
                  inter-process communication. I choose to do this by using a socket on localhost, which is actually not that slow.
                  The whole frame including MMLT tracing, neural mutation and socket communication runs at 3~4 fps. I believe with some
                  optimization we could reach some real-time frame rate.
                  The whole pipeline for each frame is shown below: 
                </p>
              </el-row>

              <el-row class="row-bg" justify="center">
                <el-col :span="12">
                  <el-image style="align:center" src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/20230325160712.png" />
                  <p><code>
                    My pipeline per frame.
                  </code></p>
                </el-col>
              </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  First we test the case that we always use large step mutation. This generally works as expected that using latent
                  neural mutation could concentrate more on important regions just as importance sampling does.
                  The figure shows how they look like after 50 frames:
                </p>
              </el-row>

              <el-row class="row-bg" justify="center">
                <el-col>
                  <div id="largestep"></div>
                  <p><code>left: Using primal mutation | right: Using latent mutation </code></p>
                </el-col>
              </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  Then we test the case that we always use small step mutation. And this turns out to be a failure. I beleive this is
                  the part that is really valuable so I think I should really spend more time figuring out why.
                  The figure shows how they look like after 200 frames:
                </p>
              </el-row>


              <el-row class="row-bg" justify="center">
                <el-col>
                  <div id="smallstep"></div>
                  <p><code>left: Using primal mutation | right: Using latent mutation </code></p>
                </el-col>
              </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  It seems that the latent mutation is more noisy but less correlated. But it is not better in acceptance rate,
                  opposite to what I expected. Using primal space mutation has acceptance rate of 0.62, while using latent space
                  mutation has acceptance rate of 0.35. It is really frustrating.
                </p>
              </el-row>

              <el-row class="row-bg" justify="center">
                <el-col :span="12">
                  <el-image style="align:center" src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/20230325184734.png" />
                  <p><code>
                    Non-optimal convergence to target distribution.
                  </code></p>
                </el-col>
              </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  I think a possible reason is by RealNVP converges to something "not regular". Here "optimal transport" might be a more accurate
                  word. We could see in practice that the mapping does not go in the optimal transport way, but result in somme strange distortion.
                  They would have quite similar distribution with the target distribution, and this is not a problem for importance sampling.
                </p>
              </el-row>

              <el-row class="row-bg" justify="center">
                <el-col :span="9">
                  <el-image style="align:center" src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/20230325184745.png" />
                  <p><code>
                    Problematic mapping.
                  </code></p>
                </el-col>
              </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  But in latent space mutation, it could be harmful. We could observe that two points very close in primal space could be 
                  mapped into two far away points in latent space. The distance / neighbor properties are not preserved well. This will
                  cause latent mutation could not traverse the neighbor region well and making things noisy.
                </p>
              </el-row>

              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  To solve this, I should try to constraint the neural network to do optimal transport.
                  There are already some works on this, like OT-Flow  <a href="#cite-otflow">[11]</a> and CP-Flow <a href="#cite-cpflow">[12]</a>.
                  But I did not have time to read them yet. Let's leave this as future work.
                </p>
              </el-row>

              <!-- Chapter 5: Survey on Mutation and Neural Mutation -->
              <el-row class="row-bg" justify="left">
                <h2 id="5.">5. Future Works</h2>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  Clearly the project is not completed yet. Here are some possible future works to do:
                </p>
                <p style="text-align:left">
                  (1). Check the transition probability whether need Jacobian. (I guess the answer is yes)<br/>
                  (2). Try more on the forward training and piecewise-polynomial coupling layer. <br/>
                  (3). Try non-neural importance mapping (like cosine-weighted or Gaussian like mapping ?). <br/>
                  (4). Try training a more powerful and efficient high-dim model. <br/>
                  (5). Learn more about the optimal-transport stuff. <br/>
                  (6). Optimize BDPT and MLT for GPU. <br/>
                  (7). Try integrating neural mutation into C++ pipeline by CUDA / Vulkan ML like things. <br/>
                  (8). ......
                </p>
              </el-row>

              <!-- Chapter 5: Survey on Mutation and Neural Mutation -->
              <!-- <el-row class="row-bg" justify="left">
                <h2 id="6.">6. Survey on Mutation and Neural Mutation</h2>
              </el-row>
              <el-row class="row-bg" justify="left">
                <h3 id="4.1" style="margin-top: 0in;">4.1 Some Theory behind Neural Mutation</h3>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  Normalizing flow is an useful technique to generate samples subject to some certain distributions by a neural network.
                  But the really interesting part that makes it different from other generative models is that it actually creates a
                  bijection between two distributions, which means the jacobian of the transformation is computable.
                </p>
                <p style="text-align:left">
                  Assume the transition kernel <math-jax latex="T_\theta"/> is defined through an implicit generative model
                  <math-jax latex="f_\theta(\cdot|x, v)"/>, where <math-jax latex=" v \sim \rho(v)"/> is an auxiliary random variable.
                </p>
                <p style="text-align:left">
                  Markov GAN (MGAN):
                  <math-jax latex="f_\theta(\cdot|x, v)"/>, where <math-jax latex=" v \sim \rho(v)"/> is an auxiliary random variable.
                </p>
              </el-row>

              <el-row class="row-bg" justify="left">
                <h3 id="4.2" style="margin-top: 0in;">4.2 Latent Space Mutation</h3>
              </el-row>
              <el-row class="row-bg" justify="left">
                <p style="text-align:left">
                  Normalizing.
                </p>
              </el-row> -->

              <!-- Reference -->
              <el-row class="row-bg" justify="left">
                <h2 id="Reference" style="text-align:left">Reference</h2>
                <ol style="padding-left: 20px;">
                  <li  id="cite-veach" style="text-align:left; list-style-type:decimal; list-style-position:outside;">
                    <a href="http://graphics.stanford.edu/papers/veach_thesis/">
                      Eric Veach. 1998. Robust monte carlo methods for light transport simulation. Ph.D. Dissertation. Stanford University, Stanford, CA, USA. Advisor(s) Leonidas J. Guibas. Order Number: AAI9837162.
                    </a>
                  </li>
                  <li  id="cite-mmlt" style="text-align:left; list-style-type:decimal; list-style-position:outside;">
                    <a href="https://dl.acm.org/doi/10.1145/2601097.2601138">
                      Toshiya Hachisuka, Anton S. Kaplanyan, and Carsten Dachsbacher. 2014. Multiplexed metropolis light transport. ACM Trans. Graph. 33, 4, Article 100 (July 2014), 10 pages. https://doi.org/10.1145/2601097.2601138
                    </a>
                  </li>
                  <li  id="cite-pbrt" style="text-align:left; list-style-type:decimal; list-style-position:outside;">
                    <a href="https://rgl.epfl.ch/software/PBRT">
                      Matt Pharr, Wenzel Jakob, and Greg Humphreys. 2016. Physically Based Rendering: From Theory to Implementation (3rd ed.). Morgan Kaufmann Publishers Inc., San Francisco, CA, USA.
                    </a>
                  </li>
                  <li  id="cite-pssmlt" style="text-align:left; list-style-type:decimal; list-style-position:outside;">
                    <a href="http://cg.iit.bme.hu/~szirmay/paper50_electronic.pdf">
                      KELEMEN, C., SZIRMAY-KALOS, L., ANTAL, G., AND CSONKA, F. 2002. A simple and robust mutation strategy for the Metropolis light transport algorithm. Computer Graphics Forum 21, 3, 531–540.
                    </a>
                  </li>
                  <li  id="cite-nice" style="text-align:left; list-style-type:decimal; list-style-position:outside;">
                    <a href="https://doi.org/10.48550/arXiv.1410.8516">
                      Laurent Dinh, David Krueger and Yoshua Bengio. 2015. NICE: Non-linear Independent Components Estimation. https://doi.org/10.48550/arXiv.1410.8516
                    </a>
                  </li>
                  <li  id="cite-realNVP" style="text-align:left; list-style-type:decimal; list-style-position:outside;">
                    <a href="https://doi.org/10.48550/arXiv.1605.08803">
                      Laurent Dinh, Jascha Sohl-Dickstein and Samy Bengio. 2017. Density estimation using Real NVP. https://doi.org/10.48550/arXiv.1605.08803
                    </a>
                  </li>
                  <li  id="cite-realNVP" style="text-align:left; list-style-type:decimal; list-style-position:outside;">
                    <a href="https://doi.org/10.48550/arXiv.1807.03039">
                      Diederik P. Kingma and Prafulla Dhariwal. 2018. Glow: Generative Flow with Invertible 1x1 Convolutions. https://doi.org/10.48550/arXiv.1807.03039
                    </a>
                  </li>
                  <li  id="cite-nis" style="text-align:left; list-style-type:decimal; list-style-position:outside;">
                    <a href="https://dl.acm.org/doi/10.1145/3341156">
                      Thomas Müller, Brian Mcwilliams, Fabrice Rousselle, Markus Gross, and Jan Novák. 2019. Neural Importance Sampling. ACM Trans. Graph. 38, 5, Article 145 (October 2019), 19 pages. https://doi.org/10.1145/3341156
                    </a>
                  </li>
                  <li  id="cite-npssis" style="text-align:left; list-style-type:decimal; list-style-position:outside;">
                    <a href="https://arxiv.org/abs/1808.07840">
                      Quan Zheng and Matthias Zwicker. 2018. Learning to Importance Sample in Primary Sample Space. https://arxiv.org/abs/1808.07840
                    </a>
                  </li>
                  <li  id="cite-ris" style="text-align:left; list-style-type:decimal; list-style-position:outside;">
                    <a href="https://dl.acm.org/doi/10.5555/2383654.2383674">
                      Justin F. Talbot, David Cline, and Parris Egbert. 2005. Importance resampling for global illumination. In Proceedings of the Sixteenth Eurographics conference on Rendering Techniques (EGSR '05). Eurographics Association, Goslar, DEU, 139–146.
                    </a>
                  </li>
                  <li  id="cite-otflow" style="text-align:left; list-style-type:decimal; list-style-position:outside;">
                    <a href="https://arxiv.org/abs/2006.00104#:~:text=OT%2DFlow%3A%20Fast%20and%20Accurate%20Continuous%20Normalizing%20Flows%20via%20Optimal%20Transport,-Derek%20Onken%2C%20Samy&text=A%20normalizing%20flow%20is%20an,density%20estimation%20and%20statistical%20inference.">
                      Derek Onken, Samy Wu Fung, Xingjian Li, and Lars Ruthotto. 2021. OT-Flow: Fast and Accurate Continuous Normalizing Flows via Optimal Transport.
                    </a>
                  </li>
                  <li  id="cite-cpflow" style="text-align:left; list-style-type:decimal; list-style-position:outside;">
                    <a href="chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fopenreview.net%2Fpdf%3Fid%3Dte7PVH1sPxJ#=&zoom=auto">
                      Chin-Wei Huang, Ricky T. Q. Chen, Christos Tsirigotis, and Aaron Courville. 2021. Convex Potential Flows: Universal Probability Distribution With Optimal Transport and Covnex Optimization.
                    </a>
                  </li>
                </ol>
              </el-row>
            </el-main>
            <el-divider />
            <el-footer>
              <el-link href="https://suikasibyl.github.io/" target="_blank" type="primary">My Homepage</el-link>
            </el-footer>
          </el-container>
        </el-container>
      </el-container>
    </el-container>
    </div>
  </div>
</template>

<style scoped>
.el-link {
  margin-right: 8px;
}
.el-link .el-icon--right.el-icon {
  vertical-align: text-bottom;
}
</style>

<script>

import '../assets/githubmd.css';
import SliderBar from 'before-after-slider'; // import

export default {
  name: 'ReportPage',
  mounted() {
    // new SliderBar({options});
    new SliderBar({
      el: '#udptvsbdpt',
      beforeImg: 'https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/udpt500small.png',
      afterImg: 'https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/bdpt500small.png'
    });
    new SliderBar({
      el: '#bdptvsmlt',
      beforeImg: 'https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/bdpt500small.png',
      afterImg: 'https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/mlt500small_fix.png'
    });
    new SliderBar({
      el: '#alldepthvsdpeth4',
      beforeImg: 'https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/depth0to3.png',
      afterImg: 'https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/depth3only.png'
    });
    new SliderBar({
      el: '#largestep',
      beforeImg: 'https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/nonneural_50frame.png',
      afterImg: 'https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/neural_50frame.png'
    });
    new SliderBar({
      el: '#smallstep',
      beforeImg: 'https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/nonneural_200small.png',
      afterImg: 'https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/neural_200small.png'
    });
  },
  props: {
  },
  data() {
    return {
      formula: '$$x = {-b \\pm \\sqrt{b^2-4ac} \\over 2a}.$$'
    }
  },
}
</script>

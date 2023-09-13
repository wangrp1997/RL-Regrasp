===================================
Spinning Up as a Deep RL Researcher
===================================
By Joshua Achiam, October 13th, 2018


.. contents:: Table of Contents
    :depth: 2

If you‚Äôre an aspiring deep RL researcher, you‚Äôve probably heard all kinds of things about deep RL by this point. You know that `it‚Äôs hard and it doesn‚Äôt always work`_. That even when you‚Äôre following a recipe, `reproducibility`_ `is a challenge`_. And that if you‚Äôre starting from scratch, `the learning curve is incredibly steep`_. It‚Äôs also the case that there are a lot of `great`_ `resources`_ `out`_ `there`_, but the material is new enough that there‚Äôs not a clear, well-charted path to mastery. The goal of this column is to help you get past the initial hurdle, and give you a clear sense of how to spin up as a deep RL researcher. In particular, this will outline a useful curriculum for increasing raw knowledge, while interleaving it with the odds and ends that lead to better research.


The Right Background
====================

**Build up a solid mathematical background.** From probability and statistics, feel comfortable with random variables, Bayes‚Äô theorem, chain rule of probability, expected values, standard deviations, and importance sampling. From multivariate calculus, understand gradients and (optionally, but it‚Äôll help) Taylor series expansions.

**Build up a general knowledge of deep learning.** You don‚Äôt need to know every single special trick and architecture, but the basics help. Know about standard architectures (`MLP`_, `vanilla RNN`_, `LSTM`_ (`also see this blog`_), `GRU`_, `conv`_ `layers`_, `resnets`_, `attention`_ `mechanisms`_), common regularizers (`weight decay`_, `dropout`_), normalization (`batch norm`_, `layer norm`_, `weight norm`_), and optimizers (`SGD, momentum SGD`_, `Adam`_, `others`_). Know what the `reparameterization trick`_ is.

**Become familiar with at least one deep learning library.** `Tensorflow`_ or `PyTorch`_ would be a good place to start. You don‚Äôt need to know how to do everything, but you should feel pretty confident in implementing a simple program to do supervised learning. 

**Get comfortable with the main concepts and terminology in RL.** Know what states, actions, trajectories, policies, rewards, value functions, and action-value functions are. If you're unfamiliar, Spinning Up ships with `an introduction`_ to this material; it's also worth checking out the `RL-Intro`_ from the OpenAI Hackathon, or the exceptional and thorough `overview by Lilian Weng`_. Optionally, if you‚Äôre the sort of person who enjoys mathematical theory, study up on the math of `monotonic improvement theory`_ (which forms the basis for advanced policy gradient algorithms), or `classical RL algorithms`_ (which despite being superseded by deep RL algorithms, contain valuable insights that sometimes drive new research).  

Learn by Doing
==============

**Write your own implementations.** You should implement as many of the core deep RL algorithms from scratch as you can, with the aim of writing the shortest correct implementation of each. This is by far the best way to develop an understanding of how they work, as well as intuitions for their specific performance characteristics. 

**Simplicity is critical.** You should organize your efforts so that you implement the simplest algorithms first, and only gradually introduce complexity. If you start off trying to build something with too many moving parts, odds are good that it will break and you'll lose weeks trying to debug it. This is a common failure mode for people who are new to deep RL, and if you find yourself stuck in it, don't be discouraged---but do try to change tack and work on a simpler algorithm instead, before returning to the more complex thing later.

**Which algorithms?** You should probably start with vanilla policy gradient (also called `REINFORCE`_), `DQN`_, `A2C`_ (the synchronous version of `A3C`_), `PPO`_ (the variant with the clipped objective), and `DDPG`_, approximately in that order. The simplest versions of all of these can be written in just a few hundred lines of code (ballpark 250-300), and some of them even less (for example, `a no-frills version of VPG`_ can be written in about 80 lines). Write single-threaded code before you try writing parallelized versions of these algorithms. (Do try to parallelize at least one.)

**Focus on understanding.** Writing working RL code requires clear, detail-oriented understanding of the algorithms. This is because **broken RL code almost always fails silently,** where the code appears to run fine except that the agent never learns how to solve the task. Usually the problem is that something is being calculated with the wrong equation, or on the wrong distribution, or data is being piped into the wrong place. Sometimes the only way to find these bugs is to read the code with a critical eye, know exactly what it should be doing, and find where it deviates from the correct behavior. Developing that knowledge requires you to engage with both academic literature and other existing implementations (when possible), so a good amount of your time should be spent on that reading. 

**What to look for in papers:** When implementing an algorithm based on a paper, scour that paper, especially the ablation analyses and supplementary material (where available). The ablations will give you an intuition for what parameters or subroutines have the biggest impact on getting things to work, which will help you diagnose bugs. Supplementary material will often give information about specific details like network architecture and optimization hyperparameters, and you should try to align your implementation to these details to improve your chances of getting it working.

**But don't overfit to paper details.** Sometimes, the paper prescribes the use of more tricks than are strictly necessary, so be a bit wary of this, and try out simplifications where possible. For example, the original DDPG paper suggests a complex neural network architecture and initialization scheme, as well as batch normalization. These aren't strictly necessary, and some of the best-reported results for DDPG use simpler networks. As another example, the original A3C paper uses asynchronous updates from the various actor-learners, but it turns out that synchronous updates work about as well. 

**Don't overfit to existing implementations either.** Study `existing`_ `implementations`_ for inspiration, but be careful not to overfit to the engineering details of those implementations. RL libraries frequently make choices for abstraction that are good for code reuse between algorithms, but which are unnecessary if you're only writing a single algorithm or supporting a single use case.

**Iterate fast in simple environments.** To debug your implementations, try them with simple environments where learning should happen quickly, like CartPole-v0, InvertedPendulum-v0, FrozenLake-v0, and HalfCheetah-v2 (with a short time horizon---only 100 or 250 steps instead of the full 1000) from the `OpenAI Gym`_. Don‚Äôt try to run an algorithm in Atari or a complex Humanoid environment if you haven‚Äôt first verified that it works on the simplest possible toy task. Your ideal experiment turnaround-time at the debug stage is <5 minutes (on your local machine) or slightly longer but not much. These small-scale experiments don't require any special hardware, and can be run without too much trouble on CPUs.  

**If it doesn't work, assume there's a bug.** Spend a lot of effort searching for bugs before you resort to tweaking hyperparameters: usually it‚Äôs a bug. Bad hyperparameters can significantly degrade RL performance, but if you're using hyperparameters similar to the ones in papers and standard implementations, those will probably not be the issue. Also worth keeping in mind: sometimes things will work in one environment even when you have a breaking bug, so make sure to test in more than one environment once your results look promising.  

**Measure everything.** Do a lot of instrumenting to see what‚Äôs going on under-the-hood. The more stats about the learning process you read out at each iteration, the easier it is to debug---after all, you can‚Äôt tell it‚Äôs broken if you can‚Äôt see that it‚Äôs breaking. I personally like to look at the mean/std/min/max for cumulative rewards, episode lengths, and value function estimates, along with the losses for the objectives, and the details of any exploration parameters (like mean entropy for stochastic policy optimization, or current epsilon for epsilon-greedy as in DQN). Also, watch videos of your agent‚Äôs performance every now and then; this will give you some insights you wouldn‚Äôt get otherwise.  

**Scale experiments when things work.** After you have an implementation of an RL algorithm that seems to work correctly in the simplest environments, test it out on harder environments. Experiments at this stage will take longer---on the order of somewhere between a few hours and a couple of days, depending. Specialized hardware---like a beefy GPU or a 32-core machine---might be useful at this point, and you should consider looking into cloud computing resources like AWS or GCE.

**Keep these habits!** These habits are worth keeping beyond the stage where you‚Äôre just learning about deep RL---they will accelerate your research!

Developing a Research Project
=============================

Once you feel reasonably comfortable with the basics in deep RL, you should start pushing on the boundaries and doing research. To get there, you'll need an idea for a project.

**Start by exploring the literature to become aware of topics in the field.** There are a wide range of topics you might find interesting: sample efficiency, exploration, transfer learning, hierarchy, memory, model-based RL, meta learning, and multi-agent, to name a few. If you're looking for inspiration, or just want to get a rough sense of what's out there, check out Spinning Up's `key papers <../spinningup/keypapers.html>`_ list. Find a paper that you enjoy on one of these subjects---something that inspires you---and read it thoroughly. Use the related work section and citations to find closely-related papers and do a deep dive in the literature. You‚Äôll start to figure out where the unsolved problems are and where you can make an impact. 

**Approaches to idea-generation:** There are a many different ways to start thinking about ideas for projects, and the frame you choose influences how the project might evolve and what risks it will face. Here are a few examples:

**Frame 1: Improving on an Existing Approach.** This is the incrementalist angle, where you try to get performance gains in an established problem setting by tweaking an existing algorithm. Reimplementing prior work is super helpful here, because it exposes you to the ways that existing algorithms are brittle and could be improved. A novice will find this the most accessible frame, but it can also be worthwhile for researchers at any level of experience. While some researchers find incrementalism less exciting, some of the most impressive achievements in machine learning have come from work of this nature.

Because projects like these are tied to existing methods, they are by nature narrowly scoped and can wrap up quickly (a few months), which may be desirable (especially when starting out as a researcher). But this also sets up the risks: it's possible that the tweaks you have in mind for an algorithm may fail to improve it, in which case, unless you come up with more tweaks, the project is just over and you have no clear signal on what to do next.

**Frame 2: Focusing on Unsolved Benchmarks.** Instead of thinking about how to improve an existing method, you aim to succeed on a task that no one has solved before. For example: achieving perfect generalization from training levels to test levels in the `Sonic domain`_ or `Gym Retro`_. When you hammer away at an unsolved task, you might try a wide variety of methods, including prior approaches and new ones that you invent for the project. It is possible for a novice to approch this kind of problem, but there will be a steeper learning curve. 

Projects in this frame have a broad scope and can go on for a while (several months to a year-plus). The main risk is that the benchmark is unsolvable without a substantial breakthrough, meaning that it would be easy to spend a lot of time without making any progress on it. But even if a project like this fails, it often leads the researcher to many new insights that become fertile soil for the next project.


**Frame 3: Create a New Problem Setting.** Instead of thinking about existing methods or current grand challenges, think of an entirely different conceptual problem that hasn't been studied yet. Then, figure out how to make progress on it. For projects along these lines, a standard benchmark probably doesn't exist yet, and you will have to design one. This can be a huge challenge, but it‚Äôs worth embracing---great benchmarks move the whole field forward.

Problems in this frame come up when they come up---it's hard to go looking for them. 

**Avoid reinventing the wheel.** When you come up with a good idea that you want to start testing, that‚Äôs great! But while you‚Äôre still in the early stages with it, do the most thorough check you can to make sure it hasn‚Äôt already been done. It can be pretty disheartening to get halfway through a project, and only then discover that there's already a paper about your idea. It's especially frustrating when the work is concurrent, which happens from time to time! But don‚Äôt let that deter you---and definitely don‚Äôt let it motivate you to plant flags with not-quite-finished research and over-claim the merits of the partial work. Do good research and finish out your projects with complete and thorough investigations, because that‚Äôs what counts, and by far what matters most in the long run.


Doing Rigorous Research in RL
=============================

Now you‚Äôve come up with an idea, and you‚Äôre fairly certain it hasn‚Äôt been done. You use the skills you‚Äôve developed to implement it and you start testing it out on standard domains. It looks like it works! But what does that mean, and how well does it have to work to be important? This is one of the hardest parts of research in deep RL. In order to validate that your proposal is a meaningful contribution, you have to rigorously prove that it actually gets a performance benefit over the strongest possible baseline algorithm---whatever currently achieves SOTA (state of the art) on your test domains. If you‚Äôve invented a new test domain, so there‚Äôs no previous SOTA, you still need to try out whatever the most reliable algorithm in the literature is that could plausibly do well in the new test domain, and then you have to beat that. 

**Set up fair comparisons.** If you implement your baseline from scratch---as opposed to comparing against another paper‚Äôs numbers directly---it‚Äôs important to spend as much time tuning your baseline as you spend tuning your own algorithm. This will make sure that comparisons are fair. Also, do your best to hold ‚Äúall else equal‚Äù even if there are substantial differences between your algorithm and the baseline. For example, if you‚Äôre investigating architecture variants, keep the number of model parameters approximately equal between your model and the baseline. Under no circumstances handicap the baseline! It turns out that the baselines in RL are pretty strong, and getting big, consistent wins over them can be tricky or require some good insight in algorithm design. 

**Remove stochasticity as a confounder.** Beware of random seeds making things look stronger or weaker than they really are, so run everything for many random seeds (at least 3, but if you want to be thorough, do 10 or more). This is really important and deserves a lot of emphasis: deep RL seems fairly brittle with respect to random seed in a lot of common use cases. There‚Äôs potentially enough variance that two different groups of random seeds can yield learning curves with differences so significant that they look like they don‚Äôt come from the same distribution at all (see `figure 10 here`_).

**Run high-integrity experiments.** Don‚Äôt just take the results from the best or most interesting runs to use in your paper. Instead, launch new, final experiments---for all of the methods that you intend to compare (if you are comparing against your own baseline implementations)---and precommit to report on whatever comes out of that. This is to enforce a weak form of `preregistration`_: you use the tuning stage to come up with your hypotheses, and you use the final runs to come up with your conclusions.

**Check each claim separately.** Another critical aspect of doing research is to run an ablation analysis. Any method you propose is likely to have several key design decisions---like architecture choices or regularization techniques, for instance---each of which could separately impact performance. The claim you'll make in your work is that those design decisions collectively help, but this is really a bundle of several claims in disguise: one for each such design element. By systematically evaluating what would happen if you were to swap them out with alternate design choices, or remove them entirely, you can figure out how to correctly attribute credit for the benefits your method confers. This lets you make each separate claim with a measure of confidence, and increases the overall strength of your work. 

Closing Thoughts
================

Deep RL is an exciting, fast-moving field, and we need as many people as possible to go through the open problems and make progress on them. Hopefully, you feel a bit more prepared to be a part of it after reading this! And whenever you‚Äôre ready, `let us know`_.

.. _`let us know`: https://jobs.lever.co/openai


PS: Other Resources
===================

Consider reading through these other informative articles about growing as a researcher or engineer in this field:

`Advice for Short-term Machine Learning Research Projects <https://rockt.github.io/2018/08/29/msc-advice>`_, by Tim Rockt√§schel, Jakob Foerster and Greg Farquhar.

`ML Engineering for AI Safety & Robustness: a Google Brain Engineer‚Äôs Guide to Entering the Field <https://80000hours.org/articles/ml-engineering-career-transition-guide/>`_, by Catherine Olsson and 80,000 Hours. 

References
==========

.. _`it‚Äôs hard and it doesn‚Äôt always work`: https://www.alexirpan.com/2018/02/14/rl-hard.html
.. [1] `Deep Reinforcement Learning Doesn't Work Yet <https://www.alexirpan.com/2018/02/14/rl-hard.html>`_, Alex Irpan, 2018

.. _`reproducibility`: https://arxiv.org/abs/1708.04133
.. _`figure 10 here`: https://arxiv.org/pdf/1708.04133.pdf
.. [2] `Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control <https://arxiv.org/abs/1708.04133>`_, Islam et al, 2017

.. _`is a challenge`: https://arxiv.org/abs/1709.06560
.. [3] `Deep Reinforcement Learning that Matters <https://arxiv.org/abs/1709.06560>`_, Henderson et al, 2017

.. _`the learning curve is incredibly steep`: http://amid.fish/reproducing-deep-rl
.. [4] `Lessons Learned Reproducing a Deep Reinforcement Learning Paper <http://amid.fish/reproducing-deep-rl>`_, Matthew Rahtz, 2018

.. _`great`: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html
.. [5] `UCL Course on RL <http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html>`_

.. _`resources`: http://rll.berkeley.edu/deeprlcourse/
.. [6] `Berkeley Deep RL Course <http://rll.berkeley.edu/deeprlcourse/>`_

.. _`out`: https://sites.google.com/view/deep-rl-bootcamp/lectures
.. [7] `Deep RL Bootcamp <https://sites.google.com/view/deep-rl-bootcamp/lectures>`_

.. _`there`: http://joschu.net/docs/nuts-and-bolts.pdf
.. [8] `Nuts and Bolts of Deep RL <http://joschu.net/docs/nuts-and-bolts.pdf>`_, John Schulman

.. _`MLP`: http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
.. [9] `Stanford Deep Learning Tutorial: Multi-Layer Neural Network <http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/>`_

.. _`Vanilla RNN`: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
.. [10] `The Unreasonable Effectiveness of Recurrent Neural Networks <http://karpathy.github.io/2015/05/21/rnn-effectiveness/>`_, Andrej Karpathy, 2015

.. _`LSTM`: https://arxiv.org/abs/1503.04069
.. [11] `LSTM: A Search Space Odyssey <https://arxiv.org/abs/1503.04069>`_, Greff et al, 2015

.. _`also see this blog`: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
.. [12] `Understanding LSTM Networks <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_, Chris Olah, 2015

.. _`GRU`: https://arxiv.org/abs/1412.3555v1
.. [13] `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling <https://arxiv.org/abs/1412.3555v1>`_, Chung et al, 2014 (GRU paper)

.. _`conv`: http://colah.github.io/posts/2014-07-Conv-Nets-Modular/
.. [14] `Conv Nets: A Modular Perspective <http://colah.github.io/posts/2014-07-Conv-Nets-Modular/>`_, Chris Olah, 2014

.. _`layers`: https://cs231n.github.io/convolutional-networks/
.. [15] `Stanford CS231n, Convolutional Neural Networks for Visual Recognition <https://cs231n.github.io/convolutional-networks/>`_

.. _`resnets`: https://arxiv.org,cb3/±512.133:5
.* [16]ahDemp#besk‰uan D%arlanc$Êor IMage ecogoitmW~!<`ttx{:/ds|i÷.org/¡Fs*9562.03;85a_<"he mt a,, û ë4 (ResFeu{)
ä,, UratventiÎn`: xttqs://!RxiV.œrg'abs/140)&0<73
.$[17]d`Ngera!ag®ine TranÛ^at	fn by Jo)ntly`LgaÚning to%A}(gÍ ·nd$ubansÏadg Jttps:/.arxkv.OÚg/ibSØ!<0;.04'1>`_8 B!(da~aU au†a,,(2p74 ®At<ention mgchanjslq)

N8_`ce„Ëaleses`;0itup{*//arhiv.orc/abs/7706,03w62
*. [1∏ﬁ& ttenkon )s Aln0IÔt Ne%d <hdtxq://ar¯iv.kÚg/i`sØ1706.0'6r>`_, VASvaFH et cl 6017
. Waweight`d%sq{`; htdˇ:/#qA0ers&Œa0s.gc/pqper-563-e≠imple%uEagjt-decay-#an-iÌ0rnva-'%nerel)8ctionn`eb
..0[1;] pArSiÌPe gaifht DEcqy C!fA…mProve UaneBcliza◊in. =httzs://paÿeR{/nyps.bc'pitgb/˜63,q-{iÈplelsemgHt}deSay≠ran-+m0rove-ganqvalkza|amN.Qƒ&>`_, Krogh and Herfz, 1990j

.> ]ddvoqo5ˆd: Ërtt2//bmLÚ*obe/paPErs.Vol}Ìd95-sravasxa&c1ta.o($.s2iva+=ava1deÆqf∆
..0Ÿ"8] `‘r_P_}t:  Q$Qhmpde ayTÔ –rmtenr Neurml$Netwo`kq f≤oÈ!Ovepfitting ,Ëdtt://jmlporg/aeZs/7onemm15/{rm~`st v!14°.i,d?sbmtastava1$e.pdf>`_$ SrkvAsvYvs et am,("214

.. _Hb!tch(norÌ`: htt1r:/a"xit"org/cbs/102.p≥∞7>. [‚ù `Ra4Ch Œgr}pl)zit)on:0AcceleratiNg!De%q Netuor+ Drairing(b"RedgcingdInvaPn!l ¡nVqrI`te Chidt"<it|8Û+.ar:iv>_rg/ab„.1µ12.8267>pO,$Kofcd aln†Sz%gedk. r01
ä.n _∞ËqYer0norm`: htTts2marxhv.oÚÁ/abs+1607.06418.. [22]"eÕa}er NopmA‰is!vim( <htpps://`Ryir.nr'qbs?127&0250.c_, Ba %t qL¨22q6

.* _@peÈght .kr≠`∫ `tDpsz//avxi6*gbf/·vs/1602.0?868.. [r3M bWuajhv ŒMpla,izxkooz†A†[hipDg RMpazamdtawiza0inn to Icse|dRave TrainÈfo of D!e0 Ãe4ral Networkq <httÙk:/a28iv.oÚg/ebs'0!2.3948>p_- samiiaNs and Ki.gÌc¨°28!6

., _ SFE, momeÓuuM SG@`: lÙtp://ufld%.;Ùcngm2$.epu‘uılrial/yupÂrv)sed/OtTimixbpimÓWtoahasty„Gr`dIenÙT}scent/ä.. [vM(‡Ct`nvObd Dee0 Âarfing \wuŒ&ia|: [tchasu)c Gradyent Des+eÆt <xttP://ug\dl.st`nford*edu?tuToriamØsua!{vkS‰d/OptimizapiOmStochaqtYcGvadi˜n4\%scenT/`]
.†µxfcm`;!httqsπØßirhivÆofg+afs'1¥1r*6880
.. [:%]!`A@cÌ> A Metbod∞fks$S|ocla34Èc OpdiiizAtikN"|`tvpb:/´arxiv,OrF/a@so;=1"*6=<p>`_, Shlwmq$`~d(B·$ 2014
..‚WPnthe2ca* Zpppc2//arxiv.r·/Aps/16p8,04607
.. [26b`An ovqrtiev(œf gratygnt descejt$opViÌm¸awionql˜k2iVhmK 4ËttRs://ARxiv.ovw/abÛ-1V∏=&04∑5>‡h Sebarlhan0BE`er-2016

. _`r'p·ra-etEr)zapiol tsick`;*ittxs+/aRxËv.grg/aby/1q2Æ6115(.m"€27’ ·AuTo-Enco‰°na Viriati/Nad(B#Yms <htDps:/Oir|iV.ore+a`s'3316.614>`_,"Kincm! and G%Ïling| 2∞33 ^upa2·metubh8at)on04rick)

n. W`dan3nrÊ,ow`2 hÙU4s:/www.t%nSovÊlo7oRG/H.. {2æ] `tensovflow`

*
$_aQyPo2cXd: `ttt://pqdOrc`/Ìg≠
?$0["9M `PyPopcH`]:
&* _`An,iÓtbgeuctionh2 .,/sˆinni.gup/ÚLﬂintRnÆhtm|*.. [0](`Spm.ning p Io Dedp(RÕ+ Hk|zodecti~†p/!RT,#PirT 1 <->/spanningur/rh_Ìnroh¸il>H_.. _qR\-Iftro : hTtps:Ø'gÈtubncnm-jacih`m'rlintro´bmob-macturØPvlsejt!ıio~/Pl_intvo.xd&
&n0[31] `SX-Intr`S!Slidu{ from OpenaM K`ck°thÌN, jOÒk Achiao,2038
n. _hovuvvaew†B{ Lih)an Uefg` Ëutps;/Olyl·aÔwÂngØivhuB.ho/lilmmogÆ2018'0"+19/a-lng,pee˚-knF//rÂIOflrceMeFÙ-learling.htmÏJ.tK32| PK hL.ng)0@Eek(i.tØ Re)nfoÚ#aieftfDfaR.k.g 8jtÙ4s://Ïil1anWenß.gitleB.io/l)lØdng/6018/"r?39/a9Ómng/pemh-(lto-rEknforÛeeaft-lear.ang/html<`| Lili‡n®WdhG, p 18
J/.(W`lOnoto*ic`imxsoveyElu"Thm/ri`:$Ë|up://foscltÆd•t?dogs/vh|siÛ.@L&
/>4[s;Z `OpÙieiZina Âyecta4kˇnÚÑ<`4u`¢?/josoËu.ngt/Ecsbheciqépdf>`_,bJOln±SÂiulcaL. 20Ò¶ wgnktnkg hm`sov5ment THÂM˙{9
>( N`cla{ci{ah ÷N(Òlgorinxmsp: lt|‡sz/'qiÙes*5alJart·.cc/|RzepÖS7aP·pgra+RlElgsInMDPÛ,0`&
*.‡Y34] `lomrithms foÚ
Re)Nfn2Cgmejt(LÂ·ívina08j|ˆps:/Smtes.ta,Íe2t`.c!^szmpes~q-|a`eru/ZLAÏgs…zMPPq,pdf>d, Csabb Sze`arvYri,"20p1 Cm`ssac`RL Anckryvjm3)
ä.* _`r'INF_“AQ`8 https2//arxiv.lr'/abs/9606.0∂7ó8
&.`{S%] obenchÔyrk-og Degp e	oForceoant dearni~g gor Kontilu}uc GOntr^| <htdps://‡rzi|.org/abq/16 4&0676:>p_,"Du!n et°al 221v*,. [`DUNÁ, ltt`s∫.Øwww*cs.|oro~to.epu/vvmvih'$#s?‰ql*xdf
Æ. [26]0‡Playi~w"Atazi 3ith$ddap PeinfnÚae-elÂ`Leazn{ng$=Ëttpq:/wWw>ccÆtoroNvÔ.e$u/~wmnih/tncs/dq~.lb>a,!Olih gp al, 201s$(D—N)
/.0_`A2C`: https2//booV.oPe<ayf„om?‚'eelile{'qbk4bmi∫c/
.†[37  KQelAI BcÒaÏioes> CCKRPc. A2C(,Idtps:/clÔg.npenai/cmm.`aselh*e{-aaKvÚa1c/&`_

/.&_`A3Gpxbhqtps2/.`pyi6.Ôcg/†cs6p:21s8w
.& €#9]†hAq˘.£`ronGu3 MDD`onÛ Fm2 D%mq RanForcemenv Learning <l¥tqs:/ar¯iv.obgkaBv/%602.9583;`Ol(Enia Et al, z4 )E3C)
J.Æ _pPXO`* hvuqs;ØØArxiv.ÔRG/kRS/1707.06343
Æ. €9] ‡Xz/ximal Podiey OpTIo	zCvio~0Al◊oritxop <xttps://ArxIr,/rG'ËÍq/q705,<2357@_,#SChullao et$`hl 2017a8p^G©
.n†YdDDP√`8pÍtxs>//arxiv.Ìrg;ebs?q50π.∞294q&o$Z60] `ÛontInuoıy CnNtrOn wivl`Dge Reinnosgemezt earniÍÁ th‘tpsz//arxiw>/rg/CbÛ?159>6≤97∫ [, Lalhicbap wt !d, 2035 ( DPG!

**_ a .o-nri(ls te"skog of VPG($htupw:/ofi§hqb>c/ÌØji„di!m/rl/mntro+blob-oaster/pgﬂ≥a6todÂ.xY
,.$[t!Y hRL/In|ro olÈa{ GpadiM.4Ra-plE CodA†<htBq://cHuhub,gom/*arjial/zh)i.ubo-b,ob/maqteb/pg]c—r|pmleÆp8<`[, josh Scji!m( 0≤8*
.* _`e|isÈo#`!(ttps>/.github.aoÌ/openIi/bawelKnÂS
.,![42]$`OPqlAI(Ba'glknas*>htxps:/-github.com/mpenai/¬asehanÂs>`_
à.. _`impl$lgn|Âtiofs`∫¿http3:O/github(commzml/rll·b
.- [43[ @bLlaf htıpr8og#)tËıb.cooorlÏ/r|Nsb>`_
"™. _`Gpeni"GyL: xttps:'/g{i&openEaNbom/
.& [04\$`open@I Gym¢<httP„∫/of˘}.penaa.com/>@_

.. W`qonic Tomıkn`:°®utus://ckntest.oqena!.aom/2018-1+
.™ [4%] pWpfnA… Re¥r/ Ck~teÛp <h|=ps:/ocon4ust.opÂlai.com/2808$1'> _
ä.. _`√˘-RetrÓ`; h4tps://‚|oe.nFdn!i*£m/g{m‹n%tro/. ö4>Y `OpgnAI Cim Ratro <htvxc://bloW.oenai.com/gxi}r}bro/>`_*N.$ _`roreekstbqtign`z!hvtps"//c_˚Èo'rZereoÎ
.. [47 `Kefter tos`Ope. bhencd <hptt3:ü.cos.Io/prereg/ø`_,(xpleiŒÎe wl¡l prezEga7|rction ·uanw mn The†#oN|gxt nf ÉcimŒt)fia tX0erimants.
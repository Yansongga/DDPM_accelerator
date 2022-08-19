

<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!--
[![Contributors][contributors-shield]][contributors-url]
#[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]-->
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Fast Diffusion Probabilistic Model Sampling through the lens of Backward Error Analysis

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

This is the official code for the paper "Fast Diffusion Probabilistic Model Sampling through the lens of Backward Error Analysis" by Baidu&Upenn.

Use the `README.md` to get started.

<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- GETTING STARTED -->
## Getting Started

It is easy to combine ys_solver_pytorch with your own pre-trained diffusion models. We support both Pytorch code. You can just copy the file `ys_solver_pytorch.py`  to your own code files and import it.

### Quick implementations

These are a few examples for using ys_solver_pytorch:
* Learning timestep schedule & Adaptive stepsize sampling 
  ```python
  from ys_solver_pytorch import ys_solver
  ## You need to firstly define your model and diffusion
  ## `model` has the format: model(x, t, **model_kwargs).
  ## If your model has no extra inputs, just let model_kwargs = {}.
  # model, diffusion = ....
  # model_kwargs = {...}
  
  my_solver = ys_solver( 
        diffusion = diffusion, 
        thres = float(args.thres),   ### You need to define your own threshold for fast sampling. You can adjust the `steps` to balance the computation costs and the sample quality.
        dpm_indices = None, 
        use_adpt = True)  ### If you would like to use adaptive step schedule sampling, make sure use_adpt = True. 
  
  
  ## 3. Sample by my_solver.sample_loop.
  sample, num_s, t_sche = my_solver.sample_loop(
        model,
        (args.batch_size, 3, args.image_size, args.image_size),    ## You need to set up your batchsize and image size. 
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        )
        
  ## "sample" is the sample images you get. 'num_s' is the average number of steps in this batch. 't_sche' is the timestep schedule in this batch
  
  
  nfe = int( args.nfe) ### This is your expected number of the forward evaluation, e.g. nfe = 8, 10, 12, 15, 20...
  t_sche = t_sche.T[:, :nfe +1])
  t_sche = t_sche[ t_sche[:, -1] == -1 ]
  t_sche = t_sche[ t_sche[:, nfe-1] >-1 ]
  t_sche = t_sche.float().mean(0).round().long().tolist()


  custom_sche = []
  for idx in range(len(t_sche)-1):
      custom_sche.append( [t_sche[idx], t_sche[idx +1]] )
  
  import torch
  torch.save( custom_sche, args.schedule_path )
  print( custom_sche, 'custom_sche' )
  print('saved ' + args.schedule_path ) 
  ```
* Customized step schedule sampling 

```python
  import torch
  from ys_solver_pytorch import ys_solver
  ## You need to firstly define your model and diffusion
  ## `model` has the format: model(x, t, **model_kwargs).
  ## If your model has no extra inputs, just let model_kwargs = {}.
  # model, diffusion = ....
  # model_kwargs = {...}
  
  my_solver = ys_solver( 
        diffusion = diffusion, 
        thres = None,   ### If you would like to use Customized step schedule sampling, there is no need to define thres. 
        dpm_indices = torch.load( args.schedule_path  ), ## You need to provide your timestep schedule saving path. 
        use_adpt = False)  ### If you would like to use Customized step schedule sampling, make sure use_adpt = False. 
  
  
  ## 3. Sample by my_solver.sample_loop.
  sample = my_solver.sample_loop(
        model,
        (args.batch_size, 3, args.image_size, args.image_size),    ## You need to set up your batchsize and image size. 
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        )
  ## "sample" is the sample images you get.
  ```
   

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Yansong Gao - [@My_homepage](https://scholar.google.com/citations?user=qxMVu4cAAAAJ&hl=en) - gaoyans@sas.upenn.edu

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS 
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)
-->
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/yansong-gao-a1aa56199/
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 

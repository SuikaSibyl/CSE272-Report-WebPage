import { createApp } from 'vue'
import ElementPlus from 'element-plus';
import MathJax, { initMathJax, renderByMathjax } from "mathjax-vue3";
import 'element-plus/dist/index.css';
import App from './App.vue'

function onMathJaxReady() {
    const el = document.getElementById("elementId");
    renderByMathjax(el);
  }
  
initMathJax({}, onMathJaxReady);
  

const app = createApp(App);
app.use(ElementPlus);
app.use(MathJax);
app.mount('#app')
pkgname=("python-sslf" "python2-sslf")
_pkgname=sslf
pkgver=0.1.0
pkgrel=1
pkgdesc="Simple spectral line-finder."
arch=("x86" "x86_64")
url="https://github.com/cjordan/sslf"
license=("MIT")
source=("git+https://github.com/cjordan/sslf")
sha256sums=('SKIP')

pkgver() {
    cd "${_pkgname}"
    grep -o '[0-9]\.[0-9]\.[0-9]' "${_pkgname}/__init__.py"
}

prepare() {
    cp -a "${_pkgname}" "${_pkgname}-py2"
}

package_python-sslf() {
    depends=("python" "python-numpy" "python-scipy")
    makedepends=("python-setuptools")

    cd "${srcdir}/${_pkgname}"
    python setup.py install \
           --optimize=1 \
           --root="${pkgdir}" \
           --prefix=/usr
}

package_python2-sslf() {
    depends=("python2" "python2-numpy" "python2-scipy")
    makedepends=("python2-setuptools")

    cd "${srcdir}/${_pkgname}-py2"
    python2 setup.py install \
            --optimize=1 \
            --root="${pkgdir}" \
            --prefix=/usr
}

package() {
    package_python
    package_python2
}
